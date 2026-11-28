#pragma once
// Presenter lock: follows ONE person through a stream of person detections.
//
// The detector reports every person it sees, several times per second, each with a box, a confidence and a small
// colour signature of their clothing. This class keeps a track of EVERY person (the way professional multi-person
// trackers do), and one of the tracks is the presenter:
//  - each track has its own motion model, so when somebody walks in front of the presenter the detection belongs to
//    that person's track and the presenter's track simply waits behind it;
//  - a detection continues the track it fits best: where that person could have moved to, about the same size, and
//    similar clothing. The presenter's clothing is remembered from several sides, so turning does not break the lock;
//  - a new person only counts after being seen several times in a row;
//  - if the presenter is lost, the camera holds. The same person is taken back when they reappear.
// The presenter is chosen automatically (the biggest person) or by the operator (request_lock).
// No dependency on OBS or on the detector, so it can be tested on its own.

#include <string>
#include <vector>

#define PT_SIG 64    // 4x4x4 colour histogram
#define PT_GALLERY 4 // views of a person's clothing that are remembered
#define PT_KP 17 // body points from the pose model: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles

struct pt_det
{
	float x0, y0, x1, y1;
	float score;
	float sig[PT_SIG];
	bool has_sig;
	float kp[PT_KP][3]; // x, y, confidence of each body point (only for the person the pose model was run on)
	bool has_kp;
};

class presenter_tracker {
public:
	enum state_e {
		st_search = 0, // nobody locked
		st_locked,     // following, confirmed by the latest detection
		st_coast,      // following on prediction, not seen for a moment
		st_lost,       // not seen for too long: holding
	};

	// settings
	float conf_new = 0.45f;  // confidence needed to start a track
	float conf_keep = 0.20f; // confidence that is enough to continue a track
	float lock_hold = 10.0f; // s. Automatic mode: how long a lost presenter keeps the lock
	float coast_time = 1.0f; // s. How long to keep predicting without a detection
	int confirm_hits = 3;    // detections in a row before a person counts
	bool manual = false;     // true: only the operator chooses the presenter, and the lock is never given away

	presenter_tracker() { reset(); }
	void reset();

	// Feed one detector result. t = time the analysed picture was taken, in seconds.
	void update(const std::vector<pt_det> &dets, double t);

	// Operator: make the person nearest to this point of the picture the presenter (at the next update).
	void request_lock(float x, float y);

	// Where the presenter is at time t. Returns false when there is nobody to follow (camera should hold).
	bool predict(double t, float &x0, float &y0, float &x1, float &y1) const;
	// How fast the presenter is moving across the picture, pixels per second (0 when nobody is being followed).
	void velocity(float &vx, float &vy) const;

	// Where the presenter's head is at time t, from the body pose. False when the pose is not known (use the top of
	// the box instead).
	bool head_point(double t, float &x, float &y) const;
	// The presenter's body points at time t (x, y, confidence), for drawing. False when not known.
	bool pose(double t, float kp[PT_KP][3]) const;
	// Which way the presenter faces: -1 = fully to the left of the picture, 0 = towards or away from the camera,
	// +1 = fully to the right. 0 when not known.
	float facing(double t) const;
	// True while a hand is held above the head (the box around the body is then taller than the person).
	bool arms_up(double t) const;

	state_e state() const;
	int people() const; // confirmed people in view
	// One line for the log each time something worth knowing happened; empty otherwise.
	std::string take_event();

private:
	struct axis_s
	{
		float p, v;       // position, speed
		float pp, pv, vv; // covariance
	};
	struct track_s
	{
		int id;
		axis_s cx, top; // centre x and top edge of the box
		float w, h;
		double t_state; // time the filter state refers to
		double t_seen;  // time of the last detection
		int hits;       // detections in a row at the start
		bool confirmed;
		float sig[PT_GALLERY][PT_SIG];
		int n_sig;
		int next_sig;
		// body pose, stored relative to the box so that it moves with the prediction
		float kp[PT_KP]
			[3]; // x as a share of the box width from its centre, y as a share of the height from its top
		double t_kp; // when the pose was last seen (0 = never)
		float head_x, head_y;
		bool has_head;
		float face_dir;
		bool hands_up;
	};
	static void ax_init(axis_s &a, float p, float r);
	static void ax_predict(axis_s &a, float dt, float q);
	static void ax_update(axis_s &a, float z, float r);
	static float similarity(const float *a, const float *b);
	static float track_similarity(const track_s &tr, const pt_det &d);

	std::vector<track_s> tracks;
	int next_id;
	int presenter_id; // 0 = nobody
	bool lost_reported;
	bool have_request;
	float req_x, req_y;
	std::string event;

	track_s *presenter();
	const track_s *presenter() const;
	void start_track(const pt_det &d, double t);
	void say(const char *what, const track_s *tr);
};
