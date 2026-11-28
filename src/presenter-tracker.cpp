#include "presenter-tracker.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

// All distances below are measured in "body heights" (the height of the presenter's box), so the same numbers
// work for a wide shot and a close shot.
static const float GATE_STILL = 0.35f; // how far a detection may be from the prediction straight away
static const float GATE_SPEED = 1.5f;  // and how much further per second without a sighting (a brisk walk)
static const float GATE_MAX = 3.0f;
static const float ACC = 3.0f;   // body heights / s^2 the motion model allows for
static const float MEAS = 0.03f; // detector box noise, body heights
// Clothing colours, 0 = nothing in common, 1 = identical. Measured on stage footage: the same person scores
// 0.68 or more against their running average, other people 0.4 or less.
static const float SIM_REJECT = 0.5f;    // below this it is somebody else, even if they stand exactly where the
					 // presenter was (somebody walking in front must not take the lock)
static const float SIM_LEARN = 0.6f;     // only clear sightings update the stored colours
static const float SIM_REACQUIRE = 0.6f; // colours this similar: the lost presenter is back

void presenter_tracker::reset()
{
	st = st_search;
	have_sig = false;
	cand_valid = false;
	cand_hits = 0;
	t_state = t_seen = 0.0;
	w = h = 0.0f;
	cx = top = axis_s{0, 0, 0, 0, 0};
	memset(sig, 0, sizeof(sig));
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
	return s; // 1 = identical colours, 0 = nothing in common
}

void presenter_tracker::advance(double t)
{
	float dt = (float)(t - t_state);
	if (dt <= 0.0f)
		return;
	if (dt > 1.0f)
		dt = 1.0f;
	ax_predict(cx, dt, ACC * h);
	ax_predict(top, dt, ACC * h);
	t_state = t;
}

void presenter_tracker::lock_on(const pt_det &d, double t, const char *why)
{
	w = d.x1 - d.x0;
	h = d.y1 - d.y0;
	ax_init(cx, (d.x0 + d.x1) * 0.5f, MEAS * h);
	ax_init(top, d.y0, MEAS * h);
	t_state = t_seen = t;
	if (d.has_sig) {
		memcpy(sig, d.sig, sizeof(sig));
		have_sig = true;
	}
	st = st_locked;
	cand_valid = false;
	cand_hits = 0;
	char b[160];
	snprintf(b, sizeof b, "presenter: %s (%dx%d px at %d,%d, confidence %.2f)", why, (int)w, (int)h, (int)d.x0,
		 (int)d.y0, d.score);
	event = b;
}

void presenter_tracker::update(const std::vector<pt_det> &dets, double t)
{
	if (st != st_search) {
		advance(t);
		const float since = (float)(t - t_seen);
		const float gate = std::min(GATE_MAX, GATE_STILL + GATE_SPEED * since);

		// the detection that is most plausibly the presenter
		int best = -1;
		float best_cost = 1e9f, best_iou = 0.0f, best_sim = 0.0f;
		const float px0 = cx.p - w * 0.5f, px1 = cx.p + w * 0.5f, py0 = top.p, py1 = top.p + h;
		for (size_t i = 0; i < dets.size(); i++) {
			const pt_det &d = dets[i];
			if (d.score < conf_keep)
				continue;
			const float dh = d.y1 - d.y0, dw = d.x1 - d.x0;
			if (dh <= 1.0f || dw <= 1.0f)
				continue;
			const float ex = ((d.x0 + d.x1) * 0.5f - cx.p) / h;
			const float ey = (d.y0 - top.p) / h;
			const float dist = sqrtf(ex * ex + ey * ey);
			if (dist > gate)
				continue;
			const float ratio = dh / h;
			if (ratio < 0.45f ||
			    ratio > 1.8f) // a lectern may hide the legs; a different person is a different size
				continue;
			const float ix = std::min(px1, d.x1) - std::max(px0, d.x0);
			const float iy = std::min(py1, d.y1) - std::max(py0, d.y0);
			const float inter = (ix > 0 && iy > 0) ? ix * iy : 0.0f;
			const float iou = inter / (w * h + dw * dh - inter);
			float sim = 1.0f;
			if (have_sig && d.has_sig) {
				sim = similarity(sig, d.sig);
				if (sim < SIM_REJECT)
					continue;
			}
			const float cost = 0.6f * dist / gate + 0.4f * (1.0f - sim);
			if (cost < best_cost) {
				best_cost = cost;
				best = (int)i;
				best_iou = iou;
				best_sim = sim;
			}
		}

		if (best >= 0) {
			const pt_det &d = dets[best];
			const float dh = d.y1 - d.y0, dw = d.x1 - d.x0;
			ax_update(cx, (d.x0 + d.x1) * 0.5f, MEAS * h);
			ax_update(top, d.y0, MEAS * h);
			w += 0.3f * (dw - w);
			h += 0.3f * (dh - h);
			if (d.has_sig && best_iou > 0.5f &&
			    (!have_sig || best_sim > SIM_LEARN)) { // colours drift slowly with the light
				if (!have_sig) {
					memcpy(sig, d.sig, sizeof(sig));
					have_sig = true;
				} else {
					for (int i = 0; i < PT_SIG; i++)
						sig[i] += 0.1f * (d.sig[i] - sig[i]);
				}
			}
			if (st != st_locked)
				event = "presenter: seen again, following";
			st = st_locked;
			t_seen = t;
			return;
		}

		// not seen this time
		if (since <= coast_time) {
			if (st == st_locked)
				st = st_coast;
			return;
		}
		if (st != st_lost) {
			st = st_lost;
			cx.v = top.v = 0.0f;
			event = "presenter: not seen, camera holds";
		}

		// The same person coming back is taken at once, wherever they are.
		if (have_sig) {
			int back = -1;
			float back_sim = SIM_REACQUIRE;
			for (size_t i = 0; i < dets.size(); i++) {
				if (dets[i].score < conf_new || !dets[i].has_sig)
					continue;
				const float sim = similarity(sig, dets[i].sig);
				if (sim > back_sim) {
					back_sim = sim;
					back = (int)i;
				}
			}
			if (back >= 0) {
				lock_on(dets[back], t, "the same person is back");
				return;
			}
		}
		if (since < lock_hold)
			return;
		// Held long enough: anybody may become the presenter now.
		event = "presenter: gone, looking for a new one";
		st = st_search;
		have_sig = false;
		cand_valid = false;
	}

	// ---- nobody locked: find a presenter. The biggest, most confident person wins, but only after being seen
	// several times in a row in about the same place.
	int best = -1;
	float best_w = 0.0f;
	for (size_t i = 0; i < dets.size(); i++) {
		const pt_det &d = dets[i];
		if (d.score < conf_new)
			continue;
		const float weight = d.score * (d.y1 - d.y0);
		if (weight > best_w) {
			best_w = weight;
			best = (int)i;
		}
	}
	if (best < 0) {
		if (cand_valid && t - cand_t > 1.5)
			cand_valid = false;
		return;
	}
	const pt_det &d = dets[best];
	const float dcx = (d.x0 + d.x1) * 0.5f, dh = d.y1 - d.y0;
	if (cand_valid && t - cand_t <= 1.5 && fabsf(dcx - cand_cx) < 0.5f * cand_h &&
	    fabsf(d.y0 - cand_top) < 0.5f * cand_h) {
		cand_hits++;
	} else {
		cand_valid = true;
		cand_hits = 1;
	}
	cand_cx = dcx;
	cand_top = d.y0;
	cand_h = dh;
	cand_t = t;
	if (cand_hits >= confirm_hits)
		lock_on(d, t, "locked on");
}

bool presenter_tracker::predict(double t, float &x0, float &y0, float &x1, float &y1) const
{
	if (st == st_search || st == st_lost)
		return false;
	// Look ahead from the last filter state, but never far: a wrong guess must not swing the camera.
	float dt = (float)(t - t_state);
	if (dt < 0.0f)
		dt = 0.0f;
	if (dt > 0.4f)
		dt = 0.4f;
	const float c = cx.p + cx.v * dt;
	const float ty = top.p + top.v * dt;
	x0 = c - w * 0.5f;
	x1 = c + w * 0.5f;
	y0 = ty;
	y1 = ty + h;
	return true;
}

std::string presenter_tracker::take_event()
{
	std::string e;
	e.swap(event);
	return e;
}
