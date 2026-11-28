#include <cmath>
#include <util/platform.h>
#include <obs-module.h>
#include "plugin-macros.generated.h"
#include "face-tracker-manager.hpp"
#include "face-detector-dlib-hog.h"
#include "face-detector-dlib-cnn.h"
#ifdef WITH_NCNN
#include "face-detector-person.h"
#endif
#include "face-tracker-dlib.h"
#include "texture-object.h"
#include "helper.hpp"

// #define debug_track(fmt, ...) blog(LOG_INFO, fmt, __VA_ARGS__)
// #define debug_detect(fmt, ...) blog(LOG_INFO, fmt, __VA_ARGS__)
#define debug_track(fmt, ...)
#define debug_detect(fmt, ...)
#define debug_track_thread(fmt, ...) // blog(LOG_INFO, fmt, __VA_ARGS__)

#define DIR_DLIB_HOG "dlib_hog_model"
#define DIR_DLIB_CNN "dlib_cnn_model"
#define DIR_DLIB_LANDMARK "dlib_face_landmark_model"
#define DIR_PERSON "person_model"

bool face_tracker_manager::person_engine_available()
{
#ifdef WITH_NCNN
	return true;
#else
	return false;
#endif
}

face_tracker_manager::face_tracker_manager()
{
	upsize_l = upsize_r = upsize_t = upsize_b = 0.0f;
	scale = 0.0f;
	tracking_threshold = 1e-2f;
	landmark_detection_data = NULL;
	crop_cur.x0 = crop_cur.x1 = crop_cur.y0 = crop_cur.y1 = 0.0f;
	tick_cnt = detect_tick = next_tick_stage_to_detector = 0;
	detector_in_progress = false;
	detect = NULL;
}

face_tracker_manager::~face_tracker_manager()
{
	for (auto &t : trackers_idlepool) {
		if (t.tracker) {
			t.tracker->stop();
			delete t.tracker;
			t.tracker = NULL;
		}
	}
	for (auto &t : trackers) {
		if (t.tracker) {
			t.tracker->stop();
			delete t.tracker;
			t.tracker = NULL;
		}
	}
	if (detect) {
		detect->stop();
		delete detect;
	}
	bfree(landmark_detection_data);
}

inline void face_tracker_manager::retire_tracker(int ix)
{
	debug_track_thread("%p retire_tracker(%d %p)", this, ix, trackers[ix].tracker);
	trackers_idlepool.push_back(trackers[ix]);
	trackers[ix].tracker->request_suspend();
	trackers.erase(trackers.begin() + ix);
}

inline bool face_tracker_manager::is_low_confident(const tracker_inst_s &t, float th1)
{
	if (t.att * t.rect.score <= th1)
		return true;

	if (t.att * t.rect.score <= tracking_threshold * t.score_first)
		return true;

	return false;
}

void face_tracker_manager::remove_duplicated_tracker()
{
	for (size_t i = 0; i < trackers.size(); i++) {
		if (trackers[i].state != tracker_inst_s::tracker_state_available)
			continue;

		rect_s r = trackers[i].rect;
		int a0 = (r.x1 - r.x0) * (r.y1 - r.y0);
		int a_overlap_sum = 0;
		bool to_remove = false;
		for (size_t j = i + 1; j < trackers.size() && !to_remove; j++) {
			if (trackers[j].state != tracker_inst_s::tracker_state_available)
				continue;
			int a = common_area(r, trackers[j].rect);
			a_overlap_sum += a;
			if (a * 10 > a0 && a_overlap_sum * 2 > a0)
				to_remove = true;
		}

		if (to_remove) {
			retire_tracker(i);
			i--;
		}
	}
}

inline void face_tracker_manager::attenuate_tracker()
{
	for (size_t i = 0; i < trackers.size(); i++) {
		if (trackers[i].state != tracker_inst_s::tracker_state_available)
			continue;
		struct tracker_inst_s &t = trackers[i];

		int a1 = (t.rect.x1 - t.rect.x0) * (t.rect.y1 - t.rect.y0);
		float amax = (float)a1 * 0.1f;
		for (size_t j = 0; j < detect_rects.size(); j++) {
			rect_s r = detect_rects[j];
			float a = (float)common_area(r, t.rect);
			if (a > amax)
				amax = a;
		}

		t.att *= powf(amax / a1, 0.1f); // if no faces, remove the tracker
	}

	float score_max = 1e-17f;
	for (size_t i = 0; i < trackers.size(); i++) {
		if (trackers[i].state == tracker_inst_s::tracker_state_available) {
			float s = trackers[i].att * trackers[i].rect.score;
			if (s > score_max)
				score_max = s;
		}
	}

	for (size_t i = 0; i < trackers.size(); i++) {
		if (trackers[i].state != tracker_inst_s::tracker_state_available)
			continue;
		if (!is_low_confident(trackers[i], 1e-2f * score_max))
			continue;

		retire_tracker(i);
		i--;
	}
}

inline void face_tracker_manager::copy_detector_to_tracker()
{
	size_t i_tracker;
	for (i_tracker = 0; i_tracker < trackers.size(); i_tracker++)
		if (trackers[i_tracker].tick_cnt == detect_tick &&
		    trackers[i_tracker].state == tracker_inst_s::tracker_state_e::tracker_state_reset_texture)
			break;
	if (i_tracker >= trackers.size())
		return;

	if (detect_rects.size() <= 0) {
		// Nothing detected. The tracker keeps following by itself, but only for 'lock_hold' seconds: after that
		// it may have drifted off the person, and a camera that holds still is better than one that wanders.
		if (target_confirmed_ns && os_gettime_ns() - target_confirmed_ns > (uint64_t)(lock_hold * 1e9f)) {
			bool any = false;
			for (size_t i = 0; i < trackers.size(); i++) {
				if (trackers[i].state == tracker_inst_s::tracker_state_available &&
				    trackers[i].att > 0.0f) {
					trackers[i].att = 0.0f;
					any = true;
				}
			}
			if (any || diag_target_state != 3)
				blog(LOG_INFO, "target: no face seen for %.0f s, released (camera holds)", lock_hold);
			diag_target_state = 3;
		}
		retire_tracker(i_tracker);
		return;
	}

	// Decide which detected face to follow. The original code always took the first one in the list, so with two
	// people in shot, or one false detection, the target could jump. Here the current target is kept: a detection
	// is used only if it overlaps the face already being tracked. If the detector cannot see the target (the person
	// turned away), the tracker carries on by itself for 'lock_hold' seconds before another face may take over.
	const uint64_t now_ns = os_gettime_ns();
	int i_target = -1;
	float s_target = 0.0f;
	for (size_t i = 0; i < trackers.size(); i++) {
		if (trackers[i].state != tracker_inst_s::tracker_state_available)
			continue;
		float s = trackers[i].att * trackers[i].rect.score;
		if (s > s_target) {
			s_target = s;
			i_target = (int)i;
		}
	}

	size_t i_largest = 0;
	for (size_t j = 1; j < detect_rects.size(); j++) {
		const rect_s &a = detect_rects[j], &b = detect_rects[i_largest];
		if ((a.x1 - a.x0) * (a.y1 - a.y0) > (b.x1 - b.x0) * (b.y1 - b.y0))
			i_largest = j;
	}

	size_t i_pick = i_largest;
	if (i_target >= 0) {
		const rect_s &tr = trackers[i_target].rect;
		int a_min = (tr.x1 - tr.x0) * (tr.y1 - tr.y0) / 10;
		int j_best = -1;
		for (size_t j = 0; j < detect_rects.size(); j++) {
			int a = common_area(detect_rects[j], tr);
			if (a > a_min) {
				a_min = a;
				j_best = (int)j;
			}
		}
		// No overlap: the person may simply have moved faster than the tracker could follow, leaving the tracker
		// sitting on the background behind them. A detected face is still the same person if it is within the
		// distance somebody can cover in the time since the target was last confirmed (about 8 face widths per
		// second, a brisk walk, plus 1.5 face widths of slack).
		int j_near = -1;
		float d_near = 0.0f;
		if (j_best < 0) {
			float tx = (tr.x0 + tr.x1) * 0.5f, ty = (tr.y0 + tr.y1) * 0.5f;
			float dt_s = target_confirmed_ns ? (float)(now_ns - target_confirmed_ns) * 1e-9f : 0.0f;
			for (size_t j = 0; j < detect_rects.size(); j++) {
				const rect_s &d = detect_rects[j];
				float dx = (d.x0 + d.x1) * 0.5f - tx, dy = (d.y0 + d.y1) * 0.5f - ty;
				float dist = sqrtf(dx * dx + dy * dy);
				float reach = (float)(d.x1 - d.x0) * (1.5f + 8.0f * dt_s);
				if (dist <= reach && (j_near < 0 || dist < d_near)) {
					j_near = (int)j;
					d_near = dist;
				}
			}
		}
		if (j_best >= 0) {
			i_pick = (size_t)j_best;
			if (diag_target_state != 1)
				blog(LOG_INFO, "target: confirmed by the detector");
			diag_target_state = 1;
		} else if (j_near >= 0) {
			i_pick = (size_t)j_near;
			blog(LOG_INFO, "target: tracker fell behind, re-attached to the face %d px away", (int)d_near);
			diag_target_state = 1;
			for (size_t i = 0; i < trackers.size(); i++) {
				if (i != i_tracker && trackers[i].state == tracker_inst_s::tracker_state_available)
					trackers[i].att = 0.0f;
			}
		} else if (now_ns - target_confirmed_ns < (uint64_t)(lock_hold * 1e9f)) {
			if (diag_target_state != 2)
				blog(LOG_INFO, "target: %d other face(s) seen, staying on the current target",
				     (int)detect_rects.size());
			diag_target_state = 2;
			retire_tracker(i_tracker);
			return;
		} else {
			blog(LOG_INFO, "target: not confirmed for %.0f s, switching to the largest face", lock_hold);
			diag_target_state = 1;
			for (size_t i = 0; i < trackers.size(); i++) {
				if (i != i_tracker && trackers[i].state == tracker_inst_s::tracker_state_available)
					trackers[i].att = 0.0f;
			}
		}
	} else {
		if (diag_target_state != 1)
			blog(LOG_INFO, "target: acquired (largest of %d face(s))", (int)detect_rects.size());
		diag_target_state = 1;
	}
	target_confirmed_ns = now_ns;

	struct tracker_inst_s &t = trackers[i_tracker];

	struct rect_s r = detect_rects[i_pick];
	int w = r.x1 - r.x0;
	int h = r.y1 - r.y0;
	r.x0 -= w * upsize_l;
	r.x1 += w * upsize_r;
	r.y0 -= h * upsize_t;
	r.y1 += h * upsize_b;
	t.tracker->set_position(r); // TODO: consider how to track two or more faces.
	t.tracker->set_upsize_info(rectf_s{upsize_l, upsize_t, upsize_r, upsize_b});
	t.tracker->start();
	t.state = tracker_inst_s::tracker_state_constructing;
}

inline void face_tracker_manager::stage_to_detector()
{
	if (!detect || detect->trylock())
		return;

	// get previous results
	if (detector_in_progress) {
		detect->get_faces(detect_rects);
		for (size_t i = 0; i < detect_rects.size(); i++)
			debug_detect("stage_to_detector: detect_rects %d %d %d %d %d %f", i, detect_rects[i].x0,
				     detect_rects[i].y0, detect_rects[i].x1, detect_rects[i].y1, detect_rects[i].score);
		// Diagnostics: report what the detector found whenever the count changes, and about every 20 s.
		if ((int)detect_rects.size() != diag_last_faces || tick_cnt - diag_last_tick > 600) {
			int ms = (int)((os_gettime_ns() - diag_detect_start_ns) / 1000000);
			if (detect_rects.size() > 0) {
				const auto &r = detect_rects[0];
				blog(LOG_INFO, "detector: %d %s; first is %dx%d px at (%d,%d) score %.2f; about %d ms",
				     (int)detect_rects.size(),
				     detector_engine == engine_person ? "person(s)" : "face(s)", r.x1 - r.x0,
				     r.y1 - r.y0, r.x0, r.y0, r.score, ms);
			} else {
				blog(LOG_INFO, "detector: no face found; took about %d ms; %d tracker(s) active", ms,
				     (int)tracker_rects.size());
			}
			diag_last_faces = (int)detect_rects.size();
			diag_last_tick = tick_cnt;
		}
		if (detector_engine == engine_person) {
#ifdef WITH_NCNN
			// The presenter lock replaces the per-face trackers: any that are left over are retired.
			for (size_t i = 0; i < trackers.size(); i++)
				trackers[i].att = 0.0f;
			attenuate_tracker();
			if (auto *d = dynamic_cast<face_detector_person *>(detect)) {
				std::vector<pt_det> persons;
				d->get_persons(persons);
				presenter.lock_hold = lock_hold;
				presenter.update(persons, presenter_det_time);
				std::string ev = presenter.take_event();
				if (!ev.empty())
					blog(LOG_INFO, "%s; detection takes about %d ms", ev.c_str(), d->last_ms());
			}
#endif
		} else {
			attenuate_tracker();
			copy_detector_to_tracker();
		}
		detector_in_progress = false;
	}

	if ((next_tick_stage_to_detector - tick_cnt) > 0) {
		detect->unlock();
		return;
	}

	if (auto cvtex = get_cvtex()) {
		detect->set_texture(cvtex, detector_crop_l, detector_crop_r, detector_crop_t, detector_crop_b);
		if (detector_engine == engine_dlib_hog) {
			if (auto *d = dynamic_cast<face_detector_dlib_hog *>(detect))
				d->set_model(detector_dlib_hog_model.c_str());
			if (auto *d = dynamic_cast<face_detector_dlib_hog *>(detect))
				d->set_upsample(detector_upsample);
		} else if (detector_engine == engine_dlib_cnn) {
			if (auto *d = dynamic_cast<face_detector_dlib_cnn *>(detect))
				d->set_model(detector_dlib_cnn_model.c_str());
		}
#ifdef WITH_NCNN
		if (detector_engine == engine_person) {
			if (auto *d = dynamic_cast<face_detector_person *>(detect)) {
				const char *name = person_model == "nano" ? "yolox_nano" : "yolox_tiny";
				char *fp = obs_module_file((std::string(DIR_PERSON "/") + name + ".param").c_str());
				char *fb = obs_module_file((std::string(DIR_PERSON "/") + name + ".bin").c_str());
				d->set_model(fp ? fp : "", fb ? fb : "");
				bfree(fp);
				bfree(fb);
			}
			presenter_det_time = (double)os_gettime_ns() * 1e-9;
		}
#endif
		detect->signal();
		diag_detect_start_ns = os_gettime_ns();
		detector_in_progress = true;
		detect_tick = tick_cnt;
		if (detector_engine == engine_person) { // no per-face tracker in this mode
			detect->unlock();
			return;
		}

		struct tracker_inst_s t;
		t.rect = rect_s{0, 0, 0, 0, 0.0f};
		t.crop_rect = rectf_s{0.0f, 0.0f, 0.0f, 0.0f};
		t.att = 0.0f;
		t.score_first = 0.0f;
		if (trackers_idlepool.size() > 0) {
			t.tracker = trackers_idlepool[0].tracker;
			trackers_idlepool[0].tracker = NULL;
			trackers_idlepool.pop_front();
		} else {
			debug_track_thread(
				"%p No available idle tracker, creating new tracker thread. There are %d existing thread.",
				this, trackers.size());
			t.tracker = new face_tracker_dlib();
			for (size_t i = 0; i < trackers.size(); i++) {
				debug_track_thread("%p existing tracker[%d]: state=%d", this, i,
						   (int)trackers[i].state);
			}
		}
		t.crop_tracker = crop_cur;
		t.state = tracker_inst_s::tracker_state_e::tracker_state_reset_texture;
		t.tick_cnt = tick_cnt;
		t.tracker->set_texture(cvtex);
		t.tracker->set_landmark_detection(landmark_detection_data);
		if (!landmark_detection_data)
			t.landmark.clear();
		trackers.push_back(t);
	}

	detect->unlock();
}

inline int face_tracker_manager::stage_surface_to_tracker(struct tracker_inst_s &t)
{
	if (auto cvtex = get_cvtex()) {
		t.tracker->set_texture(cvtex);
		t.crop_tracker = crop_cur;
		t.tracker->signal();
	} else
		return 1;
	return 0;
}

inline void face_tracker_manager::stage_to_trackers()
{
	bool have_new_tracker = false;
	for (size_t i = 0; i < trackers.size(); i++) {
		struct tracker_inst_s &t = trackers[i];
		if (t.state == tracker_inst_s::tracker_state_constructing) {
			if (!t.tracker->trylock()) {
				if (!stage_surface_to_tracker(t)) {
					t.crop_tracker = crop_cur;
					t.state = tracker_inst_s::tracker_state_first_track;
				}
				t.tracker->unlock();
				t.state = tracker_inst_s::tracker_state_first_track;
			}
		} else if (t.state == tracker_inst_s::tracker_state_first_track) {
			if (!t.tracker->trylock()) {
				bool ret = t.tracker->get_face(t.rect);
				t.crop_rect = t.crop_tracker;
				debug_track("tracker_state_first_track %p %d %d %d %d %f", t.tracker, t.rect.x0,
					    t.rect.y0, t.rect.x1, t.rect.y1, t.rect.score);
				t.att = 1.0f;
				t.score_first = t.rect.score;
				if (!ret || !landmark_detection_data || !t.tracker->get_landmark(t.landmark))
					t.landmark.resize(0);
				stage_surface_to_tracker(t);
				t.tracker->signal();
				t.tracker->unlock();
				if (ret) {
					t.state = tracker_inst_s::tracker_state_available;
					have_new_tracker = true;
				}
			}
		} else if (t.state == tracker_inst_s::tracker_state_available) {
			if (!t.tracker->trylock()) {
				bool ret = t.tracker->get_face(t.rect);
				t.crop_rect = t.crop_tracker;
				debug_track("tracker_state_available %p %d %d %d %d %f landmark=%d", t.tracker,
					    t.rect.x0, t.rect.y0, t.rect.x1, t.rect.y1, t.rect.score,
					    t.landmark.size());
				if (!ret || !landmark_detection_data || !t.tracker->get_landmark(t.landmark))
					t.landmark.resize(0);
				stage_surface_to_tracker(t);
				t.tracker->signal();
				t.tracker->unlock();
			}
		}
	}

	if (have_new_tracker)
		remove_duplicated_tracker();
}

static inline void make_tracker_rects(std::vector<face_tracker_manager::tracker_rect_s> &tracker_rects,
				      const std::deque<face_tracker_manager::tracker_inst_s> &trackers)
{
	size_t n = 0;
	for (size_t i = 0; i < trackers.size(); i++) {
		if (trackers[i].state != face_tracker_manager::tracker_inst_s::tracker_state_available)
			continue;

		float score = trackers[i].rect.score * trackers[i].att;

		if (score <= 0.0f || isnan(score))
			continue;

		if (tracker_rects.size() <= n)
			tracker_rects.resize(n + 1);
		auto &r = tracker_rects[n++];

		r.rect = trackers[i].rect;
		r.rect.score = score;
		r.crop_rect = trackers[i].crop_rect;
		r.landmark = trackers[i].landmark;
	}

	if (tracker_rects.size() > n)
		tracker_rects.resize(n);
}

void face_tracker_manager::tick(float second)
{
	if (reset_requested) {
		for (int i = trackers.size() - 1; i >= 0; i--)
			trackers[i].att = 0.0f;
		detect_rects.clear();
		presenter.reset();
		reset_requested = false;
	}

	const float interval = detector_engine == engine_person ? person_interval : detect_interval;
	if (detect_tick == tick_cnt)
		next_tick_stage_to_detector = tick_cnt + (int)(interval / second);

	tick_cnt += 1;

	if (detector_engine == engine_person) {
		// One target: the presenter, where the motion model says they are right now. The camera frames the
		// head, taken as the top of the body box.
		float x0, y0, x1, y1;
		if (presenter.predict((double)os_gettime_ns() * 1e-9, x0, y0, x1, y1)) {
			const float bw = x1 - x0, bh = y1 - y0;
			float head = std::min(bh / 7.0f, bw * 0.45f);
			if (head < 8.0f)
				head = 8.0f;
			const float cx = (x0 + x1) * 0.5f;
			tracker_rects.resize(1);
			auto &r = tracker_rects[0];
			r.rect = rect_s{(int)(cx - head * 0.5f), (int)y0, (int)(cx + head * 0.5f), (int)(y0 + head),
					1.0f};
			r.crop_rect = crop_cur;
			r.landmark.clear();
		} else {
			tracker_rects.clear();
		}
		return;
	}

	make_tracker_rects(tracker_rects, trackers);
}

void face_tracker_manager::post_render()
{
	stage_to_detector();
	stage_to_trackers();
}

static void update_detector(face_tracker_manager *ftm, enum face_tracker_manager::detector_engine_e detector_engine)
{
	if (ftm->detect) {
		ftm->detect->stop();
		delete ftm->detect;
		ftm->detect = NULL;
	}

	switch (detector_engine) {
	case face_tracker_manager::engine_dlib_hog:
		ftm->detect = new face_detector_dlib_hog();
		break;
	case face_tracker_manager::engine_dlib_cnn:
		ftm->detect = new face_detector_dlib_cnn();
		break;
	case face_tracker_manager::engine_person:
#ifdef WITH_NCNN
		ftm->detect = new face_detector_person();
#else
		blog(LOG_WARNING, "person detection is not part of this build; using the face detector instead");
		ftm->detect = new face_detector_dlib_hog();
		detector_engine = face_tracker_manager::engine_dlib_hog;
#endif
		break;
	default:
		blog(LOG_ERROR, "unknown detector_engine %d", (int)detector_engine);
	}

	ftm->detector_engine = detector_engine;
	ftm->presenter.reset();
	ftm->reset_requested = true;

	if (ftm->detect)
		ftm->detect->start();
}

void face_tracker_manager::update(obs_data_t *settings)
{
	upsize_l = obs_data_get_double(settings, "upsize_l");
	upsize_r = obs_data_get_double(settings, "upsize_r");
	upsize_t = obs_data_get_double(settings, "upsize_t");
	upsize_b = obs_data_get_double(settings, "upsize_b");
	scale = obs_data_get_double(settings, "scale");
	auto _detector_engine = (enum detector_engine_e)obs_data_get_int(settings, "detector_engine");
	if (_detector_engine == engine_person && !person_engine_available())
		_detector_engine = engine_dlib_hog;
	if (_detector_engine != detector_engine)
		update_detector(this, _detector_engine);
	person_model = obs_data_get_string(settings, "person_model");
	person_interval = (float)obs_data_get_double(settings, "person_interval");
	if (person_interval < 0.05f)
		person_interval = 0.05f;
	detector_dlib_hog_model = obs_data_get_string(settings, "detector_dlib_hog_model");
	detector_dlib_cnn_model = obs_data_get_string(settings, "detector_dlib_cnn_model");
	detector_crop_l = obs_data_get_int(settings, "detector_crop_l");
	detector_crop_r = obs_data_get_int(settings, "detector_crop_r");
	detector_crop_t = obs_data_get_int(settings, "detector_crop_t");
	detector_crop_b = obs_data_get_int(settings, "detector_crop_b");
	detector_upsample = obs_data_get_bool(settings, "detector_upsample");
	detect_interval = (float)obs_data_get_double(settings, "detect_interval");
	if (detect_interval < 0.2f)
		detect_interval = 0.2f;
	lock_hold = (float)obs_data_get_double(settings, "lock_hold");
	if (lock_hold < 1.0f)
		lock_hold = 1.0f;
	bool landmark_detection = obs_data_get_bool(settings, "landmark_detection");
	bfree(landmark_detection_data);
	landmark_detection_data = NULL;
	if (landmark_detection)
		landmark_detection_data = bstrdup(obs_data_get_string(settings, "landmark_detection_data"));
	if (obs_data_get_bool(settings, "tracking_th_en"))
		tracking_threshold = from_dB(obs_data_get_double(settings, "tracking_th_dB"));
	else
		tracking_threshold = 0.0;
}

static bool tracking_th_en_modified(obs_properties_t *props, obs_property_t *, obs_data_t *settings)
{
	bool tracking_th_en = obs_data_get_bool(settings, "tracking_th_en");
	obs_property_t *tracking_th_dB = obs_properties_get(props, "tracking_th_dB");
	obs_property_set_visible(tracking_th_dB, tracking_th_en);
	return true;
}

void face_tracker_manager::get_properties(obs_properties_t *pp)
{
	obs_property_t *p;
	std::string data_path = obs_get_module_data_path(obs_current_module());

	obs_properties_add_float(pp, "upsize_l", obs_module_text("Face box padding: left"), -0.4, 4.0, 0.2);
	obs_properties_add_float(pp, "upsize_r", obs_module_text("Face box padding: right"), -0.4, 4.0, 0.2);
	obs_properties_add_float(pp, "upsize_t", obs_module_text("Face box padding: top"), -0.4, 4.0, 0.2);
	obs_properties_add_float(pp, "upsize_b", obs_module_text("Face box padding: bottom"), -0.4, 4.0, 0.2);
	obs_properties_add_float(
		pp, "scale", obs_module_text("Shrink picture before analysing (1 = finds smaller faces, 2 = less CPU)"),
		1.0, 16.0, 1.0);
	p = obs_properties_add_list(pp, "detector_engine", obs_module_text("Detector"), OBS_COMBO_TYPE_LIST,
				    OBS_COMBO_FORMAT_INT);
	if (person_engine_available())
		obs_property_list_add_int(p, obs_module_text("Whole person (recommended)"), (int)engine_person);
	obs_property_list_add_int(p, obs_module_text("Detector.dlib.hog"), (int)engine_dlib_hog);
	obs_property_list_add_int(p, obs_module_text("Detector.dlib.cnn"), (int)engine_dlib_cnn);
	obs_properties_add_path(pp, "detector_dlib_hog_model", obs_module_text("Dlib HOG model"), OBS_PATH_FILE,
				"Data Files (*.dat);;"
				"All Files (*.*)",
				(data_path + "/" DIR_DLIB_CNN).c_str());
	obs_properties_add_path(pp, "detector_dlib_cnn_model", obs_module_text("Dlib CNN model"), OBS_PATH_FILE,
				"Data Files (*.dat);;"
				"All Files (*.*)",
				(data_path + "/" DIR_DLIB_CNN).c_str());
	p = obs_properties_add_int(pp, "detector_crop_l", obs_module_text("Search area: ignore this much on the left"),
				   0, 1920, 1);
	obs_property_int_set_suffix(p, " px");
	p = obs_properties_add_int(pp, "detector_crop_r", obs_module_text("Search area: ignore this much on the right"),
				   0, 1920, 1);
	obs_property_int_set_suffix(p, " px");
	p = obs_properties_add_int(pp, "detector_crop_t", obs_module_text("Search area: ignore this much at the top"),
				   0, 1080, 1);
	obs_property_int_set_suffix(p, " px");
	p = obs_properties_add_int(pp, "detector_crop_b",
				   obs_module_text("Search area: ignore this much at the bottom"), 0, 1080, 1);
	obs_property_int_set_suffix(p, " px");
	p = obs_properties_add_list(pp, "person_model", obs_module_text("Person detection quality"),
				    OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(p, obs_module_text("Accurate (default)"), "tiny");
	obs_property_list_add_string(p, obs_module_text("Fast (weaker PC, bigger person in the picture)"), "nano");
	p = obs_properties_add_float(pp, "person_interval", obs_module_text("Look for the person every"), 0.05, 1.0,
				     0.05);
	obs_property_float_set_suffix(p, " s");
	obs_properties_add_bool(pp, "detector_upsample", obs_module_text("Find small faces (HOG only, more CPU)"));
	p = obs_properties_add_float(pp, "detect_interval", obs_module_text("Detect every"), 0.2, 5.0, 0.1);
	obs_property_float_set_suffix(p, " s");
	p = obs_properties_add_float(pp, "lock_hold",
				     obs_module_text("Stay on the same person when the face is hidden, for"), 1.0, 60.0,
				     1.0);
	obs_property_float_set_suffix(p, " s");
	obs_properties_add_bool(pp, "landmark_detection", obs_module_text("Enable landmark detection"));
	p = obs_properties_add_path(pp, "landmark_detection_data", obs_module_text("Landmark detection data"),
				    OBS_PATH_FILE,
				    "Data Files (*.dat);;"
				    "All Files (*.*)",
				    (data_path + "/" DIR_DLIB_LANDMARK).c_str());
	obs_property_set_long_description(
		p, obs_module_text("You can get the shape_predictor_68_face_landmarks.dat file from: "
				   "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"));
	p = obs_properties_add_bool(pp, "tracking_th_en", obs_module_text("Set tracking threshold"));
	obs_property_set_modified_callback(p, tracking_th_en_modified);
	p = obs_properties_add_float(pp, "tracking_th_dB", obs_module_text("Tracking threshold"), -120.0, -20.0, 5.0);
	obs_property_float_set_suffix(p, " dB");
}

void face_tracker_manager::get_defaults(obs_data_t *settings)
{
	obs_data_set_default_double(settings, "upsize_l", 0.2);
	obs_data_set_default_double(settings, "upsize_r", 0.2);
	obs_data_set_default_double(settings, "upsize_t", 0.3);
	obs_data_set_default_double(settings, "upsize_b", 0.1);
	obs_data_set_default_double(settings, "scale", 2.0);
	obs_data_set_default_double(settings, "detect_interval", 1.0);
	obs_data_set_default_double(settings, "lock_hold", 6.0);
	obs_data_set_default_string(settings, "person_model", "tiny");
	obs_data_set_default_double(settings, "person_interval", 0.15);
	obs_data_set_default_bool(settings, "tracking_th_en", true);
	obs_data_set_default_double(settings, "tracking_th_dB", -80.0);

	if (char *f = obs_module_file(DIR_DLIB_HOG "/frontal_face_detector.dat")) {
		obs_data_set_default_string(settings, "detector_dlib_hog_model", f);
		bfree(f);
	} else {
		blog(LOG_ERROR, "frontal_face_detector.dat is not found in the data directory.");
	}

	if (char *f = obs_module_file(DIR_DLIB_CNN "/mmod_human_face_detector.dat")) {
		obs_data_set_default_string(settings, "detector_dlib_cnn_model", f);
		bfree(f);
	} else {
		blog(LOG_ERROR, "mmod_human_face_detector.dat is not found in the data directory.");
	}

	if (char *f = obs_module_file(DIR_DLIB_LANDMARK "/shape_predictor_5_face_landmarks.dat")) {
		obs_data_set_default_string(settings, "landmark_detection_data", f);
		bfree(f);
	} else {
		blog(LOG_ERROR, "shape_predictor_5_face_landmarks.dat is not found in the data directory.");
	}
}
