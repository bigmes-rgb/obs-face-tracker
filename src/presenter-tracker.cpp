#include "presenter-tracker.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

// All distances below are measured in "body heights" (the height of the person's box), so the same numbers work
// for a wide shot and a close shot.
static const float GATE_STILL = 0.35f; // how far a detection may be from the prediction straight away
static const float GATE_SPEED = 1.5f;  // and how much further per second without a sighting (a brisk walk)
static const float GATE_MAX = 3.0f;
static const float ACC = 3.0f;    // body heights / s^2 the motion model allows for
static const float MEAS = 0.03f;  // detector box noise, body heights
static const float FORGET = 1.5f; // s. Other people are forgotten this long after they were last seen
// Clothing colours, 0 = nothing in common, 1 = identical. Measured on stage footage: the same person scores 0.68 or
// more against their running average, other people 0.4 or less.
static const float SIM_SAME = 0.6f;     // this similar: the same person (used to take a lost presenter back)
static const float SIM_NEW_VIEW = 0.7f; // below this against every remembered view: remember this view as well

void presenter_tracker::reset()
{
	tracks.clear();
	next_id = 1;
	presenter_id = 0;
	lost_reported = false;
	have_request = false;
	req_x = req_y = 0.0f;
}

void presenter_tracker::ax_init(axis_s &a, float p, float r)
{
	a.p = p;
	a.v = 0.0f;
	a.pp = r * r;
	a.pv = 0.0f;
	a.vv = 0.0f;
}

void presenter_tracker::ax_predict(axis_s &a, float dt, float q)
{
	// constant-velocity model driven by random acceleration of standard deviation q
	a.p += a.v * dt;
	const float dt2 = dt * dt;
	a.pp += dt * (2.0f * a.pv + dt * a.vv) + q * q * dt2 * dt2 * 0.25f;
	a.pv += dt * a.vv + q * q * dt2 * dt * 0.5f;
	a.vv += q * q * dt2;
}

void presenter_tracker::ax_update(axis_s &a, float z, float r)
{
	const float s = a.pp + r * r;
	const float kp = a.pp / s, kv = a.pv / s;
	const float y = z - a.p;
	a.p += kp * y;
	a.v += kv * y;
	const float pp = a.pp, pv = a.pv;
	a.pp = (1.0f - kp) * pp;
	a.pv = (1.0f - kp) * pv;
	a.vv -= kv * pv;
}

float presenter_tracker::similarity(const float *a, const float *b)
{
	float s = 0.0f;
	for (int i = 0; i < PT_SIG; i++)
		s += std::min(a[i], b[i]);
	return s;
}

// How well the clothing of a detection matches a track: the best of the remembered views. 1 when unknown.
float presenter_tracker::track_similarity(const track_s &tr, const pt_det &d)
{
	if (!d.has_sig || tr.n_sig == 0)
		return 1.0f;
	float best = 0.0f;
	for (int i = 0; i < tr.n_sig; i++)
		best = std::max(best, similarity(tr.sig[i], d.sig));
	return best;
}

presenter_tracker::track_s *presenter_tracker::presenter()
{
	for (size_t i = 0; i < tracks.size(); i++)
		if (tracks[i].id == presenter_id)
			return &tracks[i];
	return nullptr;
}

const presenter_tracker::track_s *presenter_tracker::presenter() const
{
	for (size_t i = 0; i < tracks.size(); i++)
		if (tracks[i].id == presenter_id)
			return &tracks[i];
	return nullptr;
}

void presenter_tracker::start_track(const pt_det &d, double t)
{
	track_s tr;
	tr.id = next_id++;
	tr.w = d.x1 - d.x0;
	tr.h = d.y1 - d.y0;
	ax_init(tr.cx, (d.x0 + d.x1) * 0.5f, MEAS * tr.h);
	ax_init(tr.top, d.y0, MEAS * tr.h);
	tr.t_state = tr.t_seen = t;
	tr.hits = 1;
	tr.confirmed = false;
	tr.n_sig = 0;
	tr.next_sig = 1;
	memset(tr.sig, 0, sizeof(tr.sig));
	memset(tr.kp, 0, sizeof(tr.kp));
	tr.t_kp = 0.0;
	tr.head_x = tr.head_y = 0.0f;
	tr.has_head = false;
	tr.face_dir = 0.0f;
	tr.hands_up = false;
	if (d.has_sig) {
		memcpy(tr.sig[0], d.sig, sizeof(tr.sig[0]));
		tr.n_sig = 1;
	}
	tracks.push_back(tr);
}

void presenter_tracker::say(const char *what, const track_s *tr)
{
	char b[200];
	if (tr)
		snprintf(b, sizeof b, "presenter: %s (person %d, %dx%d px at %d,%d; %d people in view)", what, tr->id,
			 (int)tr->w, (int)tr->h, (int)(tr->cx.p - tr->w * 0.5f), (int)tr->top.p, people());
	else
		snprintf(b, sizeof b, "presenter: %s (%d people in view)", what, people());
	event = b;
}

void presenter_tracker::request_lock(float x, float y)
{
	have_request = true;
	req_x = x;
	req_y = y;
}

void presenter_tracker::update(const std::vector<pt_det> &dets, double t)
{
	// 1. move every track forward to the time of this picture
	for (size_t i = 0; i < tracks.size(); i++) {
		track_s &tr = tracks[i];
		float dt = (float)(t - tr.t_state);
		if (dt > 0.0f) {
			if (dt > 1.0f)
				dt = 1.0f;
			// a track that has not been seen for a while stops where it is
			if (t - tr.t_seen > coast_time)
				tr.cx.v = tr.top.v = 0.0f;
			ax_predict(tr.cx, dt, ACC * tr.h);
			ax_predict(tr.top, dt, ACC * tr.h);
			tr.t_state = t;
		}
	}

	// 2. give each detection to the track it fits best (cheapest pairs first)
	struct pair_s
	{
		float cost;
		int ti, di;
		float iou, sim;
	};
	std::vector<pair_s> pairs;
	for (size_t ti = 0; ti < tracks.size(); ti++) {
		const track_s &tr = tracks[ti];
		const float since = (float)(t - tr.t_seen);
		const float gate = std::min(GATE_MAX, GATE_STILL + GATE_SPEED * since);
		const float px0 = tr.cx.p - tr.w * 0.5f, px1 = tr.cx.p + tr.w * 0.5f;
		const float py0 = tr.top.p, py1 = tr.top.p + tr.h;
		for (size_t di = 0; di < dets.size(); di++) {
			const pt_det &d = dets[di];
			if (d.score < conf_keep)
				continue;
			const float dh = d.y1 - d.y0, dw = d.x1 - d.x0;
			if (dh <= 1.0f || dw <= 1.0f)
				continue;
			const float ex = ((d.x0 + d.x1) * 0.5f - tr.cx.p) / tr.h;
			const float ey = (d.y0 - tr.top.p) / tr.h;
			const float dist = sqrtf(ex * ex + ey * ey);
			if (dist > gate)
				continue;
			const float ratio = dh / tr.h;
			if (ratio < 0.45f ||
			    ratio > 1.8f) // a lectern may hide the legs; another person is another size
				continue;
			const float ix = std::min(px1, d.x1) - std::max(px0, d.x0);
			const float iy = std::min(py1, d.y1) - std::max(py0, d.y0);
			const float inter = (ix > 0 && iy > 0) ? ix * iy : 0.0f;
			pair_s p;
			p.iou = inter / (tr.w * tr.h + dw * dh - inter);
			p.sim = track_similarity(tr, d);
			p.cost = 0.5f * dist / gate + 0.5f * (1.0f - p.sim);
			p.ti = (int)ti;
			p.di = (int)di;
			pairs.push_back(p);
		}
	}
	std::sort(pairs.begin(), pairs.end(), [](const pair_s &a, const pair_s &b) { return a.cost < b.cost; });
	std::vector<char> track_used(tracks.size(), 0), det_used(dets.size(), 0);
	for (size_t k = 0; k < pairs.size(); k++) {
		const pair_s &p = pairs[k];
		if (track_used[p.ti] || det_used[p.di])
			continue;
		track_used[p.ti] = det_used[p.di] = 1;
		track_s &tr = tracks[p.ti];
		const pt_det &d = dets[p.di];
		ax_update(tr.cx, (d.x0 + d.x1) * 0.5f, MEAS * tr.h);
		ax_update(tr.top, d.y0, MEAS * tr.h);
		tr.w += 0.3f * ((d.x1 - d.x0) - tr.w);
		tr.h += 0.3f * ((d.y1 - d.y0) - tr.h);
		tr.t_seen = t;
		if (!tr.confirmed && ++tr.hits >= confirm_hits)
			tr.confirmed = true;

		// Body pose (when the pose model was run on this person): remember it relative to the box. The head is
		// the middle of the nose, eyes and ears that were found with confidence.
		if (d.has_kp) {
			const float bw = d.x1 - d.x0, bh = d.y1 - d.y0, bcx = (d.x0 + d.x1) * 0.5f;
			for (int k = 0; k < PT_KP; k++) {
				tr.kp[k][0] = (d.kp[k][0] - bcx) / bw;
				tr.kp[k][1] = (d.kp[k][1] - d.y0) / bh;
				tr.kp[k][2] = d.kp[k][2];
			}
			tr.t_kp = t;
			float sx = 0.0f, sy = 0.0f;
			int n = 0;
			for (int k = 0; k < 5; k++) {
				if (tr.kp[k][2] >= 0.35f && tr.kp[k][1] > -0.1f && tr.kp[k][1] < 0.5f &&
				    fabsf(tr.kp[k][0]) < 0.6f) {
					sx += tr.kp[k][0];
					sy += tr.kp[k][1];
					n++;
				}
			}
			if (n >= 3) {
				const float hx = sx / (float)n, hy = sy / (float)n;
				if (!tr.has_head) {
					tr.head_x = hx;
					tr.head_y = hy;
					tr.has_head = true;
				} else {
					tr.head_x += 0.4f * (hx - tr.head_x);
					tr.head_y += 0.4f * (hy - tr.head_y);
				}
				// A hand above the head makes the box taller than the person.
				tr.hands_up = false;
				for (int k = 9; k <= 10; k++)
					if (tr.kp[k][2] >= 0.35f && tr.kp[k][1] < hy)
						tr.hands_up = true;
				// Facing, from the face itself: seen from the front the nose is midway between the
				// ears; turned to a side it moves towards, and then past, the ear on that side.
				const bool el = tr.kp[3][2] >= 0.35f, er = tr.kp[4][2] >= 0.35f;
				if (tr.kp[0][2] >= 0.35f && (el || er)) {
					float f;
					if (el && er) {
						const float mid = (tr.kp[3][0] + tr.kp[4][0]) * 0.5f;
						const float half =
							std::max(0.01f, fabsf(tr.kp[3][0] - tr.kp[4][0]) * 0.5f);
						f = (tr.kp[0][0] - mid) / half;
					} else {
						f = tr.kp[0][0] > tr.kp[el ? 3 : 4][0] ? 1.0f : -1.0f;
					}
					f = std::max(-1.0f, std::min(1.0f, f));
					tr.face_dir += 0.25f * (f - tr.face_dir);
				}
			}
		}

		// Clothing: follow slow changes of the best-matching view, and remember a new view when the person has
		// turned. Only while nobody else is close, so that somebody passing in front is never learned.
		if (d.has_sig && p.iou > 0.5f) {
			bool alone = true;
			for (size_t j = 0; j < tracks.size() && alone; j++) {
				if ((int)j == p.ti)
					continue;
				alone = fabsf(tracks[j].cx.p - tr.cx.p) > (tracks[j].w + tr.w) * 0.75f;
			}
			if (tr.n_sig == 0) {
				memcpy(tr.sig[0], d.sig, sizeof(tr.sig[0]));
				tr.n_sig = 1;
			} else if (alone) {
				int best = 0;
				float best_sim = -1.0f;
				for (int i = 0; i < tr.n_sig; i++) {
					const float s = similarity(tr.sig[i], d.sig);
					if (s > best_sim) {
						best_sim = s;
						best = i;
					}
				}
				if (best_sim < SIM_NEW_VIEW) {
					const int slot = tr.n_sig < PT_GALLERY ? tr.n_sig++ : tr.next_sig;
					memcpy(tr.sig[slot], d.sig, sizeof(tr.sig[slot]));
					tr.next_sig = 1 + (tr.next_sig % (PT_GALLERY - 1)); // the first view is kept
				} else {
					for (int i = 0; i < PT_SIG; i++)
						tr.sig[best][i] += 0.1f * (d.sig[i] - tr.sig[best][i]);
				}
			}
		}
	}

	// 3. a lost presenter who is back somewhere else: recognised by the clothing
	track_s *pr = presenter();
	if (pr && t - pr->t_seen > coast_time && pr->n_sig > 0) {
		int back = -1;
		float back_sim = SIM_SAME;
		for (size_t di = 0; di < dets.size(); di++) {
			if (det_used[di] || dets[di].score < conf_new || !dets[di].has_sig)
				continue;
			const float sim = track_similarity(*pr, dets[di]);
			if (sim > back_sim) {
				back_sim = sim;
				back = (int)di;
			}
		}
		if (back >= 0) {
			const pt_det &d = dets[back];
			det_used[back] = 1;
			pr->w = d.x1 - d.x0;
			pr->h = d.y1 - d.y0;
			ax_init(pr->cx, (d.x0 + d.x1) * 0.5f, MEAS * pr->h);
			ax_init(pr->top, d.y0, MEAS * pr->h);
			pr->t_state = pr->t_seen = t;
		}
	}

	// ...or who has already been picked up as a "new" person: that track becomes the presenter's
	pr = presenter();
	if (pr && t - pr->t_seen > coast_time && pr->n_sig > 0) {
		int back = -1;
		float back_sim = SIM_SAME;
		for (size_t i = 0; i < tracks.size(); i++) {
			const track_s &o = tracks[i];
			if (o.id == presenter_id || o.n_sig == 0 || t - o.t_seen > 0.3)
				continue;
			float sim = 0.0f;
			for (int a = 0; a < pr->n_sig; a++)
				for (int b = 0; b < o.n_sig; b++)
					sim = std::max(sim, similarity(pr->sig[a], o.sig[b]));
			if (sim > back_sim) {
				back_sim = sim;
				back = (int)i;
			}
		}
		if (back >= 0) {
			const track_s o = tracks[back];
			pr->cx = o.cx;
			pr->top = o.top;
			pr->w = o.w;
			pr->h = o.h;
			pr->t_state = o.t_state;
			pr->t_seen = o.t_seen;
			tracks.erase(tracks.begin() + back);
		}
	}

	// 4. detections nobody claimed are new people
	for (size_t di = 0; di < dets.size(); di++) {
		if (!det_used[di] && dets[di].score >= conf_new)
			start_track(dets[di], t);
	}

	// 5. forget people who have gone. The presenter is kept: for lock_hold seconds in automatic mode, and until
	// the operator says otherwise in manual mode.
	for (size_t i = 0; i < tracks.size();) {
		const track_s &tr = tracks[i];
		const float since = (float)(t - tr.t_seen);
		bool drop = since > FORGET || (!tr.confirmed && since > 0.5f);
		if (tr.id == presenter_id)
			drop = !manual && since > lock_hold;
		if (drop) {
			const bool was_presenter = tr.id == presenter_id;
			tracks.erase(tracks.begin() + i);
			if (was_presenter) {
				presenter_id = 0;
				say("gone, looking for a new one", nullptr);
			}
		} else {
			i++;
		}
	}

	// 6. who is the presenter?
	if (have_request) {
		have_request = false;
		const track_s *best = nullptr;
		float best_d = 1e18f;
		for (size_t i = 0; i < tracks.size(); i++) {
			const track_s &tr = tracks[i];
			if (t - tr.t_seen > 0.5)
				continue;
			// distance from the requested point to the person's upper body
			const float dx = tr.cx.p - req_x, dy = (tr.top.p + 0.25f * tr.h) - req_y;
			const float d2 = dx * dx + dy * dy;
			if (d2 < best_d) {
				best_d = d2;
				best = &tr;
			}
		}
		if (best) {
			presenter_id = best->id;
			lost_reported = false;
			say("locked by the operator", best);
		} else {
			say("lock requested, but nobody is in view", nullptr);
		}
	} else if (presenter_id == 0 && !manual) {
		const track_s *best = nullptr;
		float best_h = 0.0f;
		for (size_t i = 0; i < tracks.size(); i++) {
			const track_s &tr = tracks[i];
			if (!tr.confirmed || t - tr.t_seen > 0.3)
				continue;
			if (tr.h > best_h) {
				best_h = tr.h;
				best = &tr;
			}
		}
		if (best) {
			presenter_id = best->id;
			lost_reported = false;
			say("locked on", best);
		}
	}

	// 7. report losing and finding the presenter
	pr = presenter();
	if (pr) {
		const bool lost = t - pr->t_seen > coast_time;
		if (lost && !lost_reported)
			say("not seen, camera holds", pr);
		else if (!lost && lost_reported)
			say("seen again, following", pr);
		lost_reported = lost;
	}
}

presenter_tracker::state_e presenter_tracker::state() const
{
	const track_s *pr = presenter();
	if (!pr)
		return st_search;
	const double since = pr->t_state - pr->t_seen;
	if (since > coast_time)
		return st_lost;
	return since > 0.0 ? st_coast : st_locked;
}

int presenter_tracker::people() const
{
	int n = 0;
	for (size_t i = 0; i < tracks.size(); i++)
		if (tracks[i].confirmed)
			n++;
	return n;
}

bool presenter_tracker::predict(double t, float &x0, float &y0, float &x1, float &y1) const
{
	const track_s *pr = presenter();
	if (!pr || pr->t_state - pr->t_seen > coast_time)
		return false;
	// Look ahead from the last filter state, but never far: a wrong guess must not swing the camera.
	float dt = (float)(t - pr->t_state);
	if (dt < 0.0f)
		dt = 0.0f;
	if (dt > 0.4f)
		dt = 0.4f;
	const float c = pr->cx.p + pr->cx.v * dt;
	const float ty = pr->top.p + pr->top.v * dt;
	x0 = c - pr->w * 0.5f;
	x1 = c + pr->w * 0.5f;
	y0 = ty;
	y1 = ty + pr->h;
	return true;
}

bool presenter_tracker::head_point(double t, float &x, float &y) const
{
	const track_s *pr = presenter();
	float x0, y0, x1, y1;
	if (!pr || !pr->has_head || t - pr->t_kp > 1.5 || !predict(t, x0, y0, x1, y1))
		return false;
	x = (x0 + x1) * 0.5f + pr->head_x * (x1 - x0);
	y = y0 + pr->head_y * (y1 - y0);
	return true;
}

bool presenter_tracker::pose(double t, float kp[PT_KP][3]) const
{
	const track_s *pr = presenter();
	float x0, y0, x1, y1;
	if (!pr || pr->t_kp == 0.0 || t - pr->t_kp > 1.0 || !predict(t, x0, y0, x1, y1))
		return false;
	for (int k = 0; k < PT_KP; k++) {
		kp[k][0] = (x0 + x1) * 0.5f + pr->kp[k][0] * (x1 - x0);
		kp[k][1] = y0 + pr->kp[k][1] * (y1 - y0);
		kp[k][2] = pr->kp[k][2];
	}
	return true;
}

float presenter_tracker::facing(double t) const
{
	const track_s *pr = presenter();
	if (!pr || !pr->has_head || t - pr->t_kp > 1.5)
		return 0.0f;
	return pr->face_dir;
}

bool presenter_tracker::arms_up(double t) const
{
	const track_s *pr = presenter();
	return pr && pr->has_head && t - pr->t_kp < 1.5 && pr->hands_up;
}

void presenter_tracker::velocity(float &vx, float &vy) const
{
	const track_s *pr = presenter();
	if (pr && pr->t_state - pr->t_seen <= coast_time) {
		vx = pr->cx.v;
		vy = pr->top.v;
	} else {
		vx = vy = 0.0f;
	}
}

std::string presenter_tracker::take_event()
{
	std::string e;
	e.swap(event);
	return e;
}
