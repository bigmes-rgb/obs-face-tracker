#pragma once
// Presenter lock: follows ONE person through a stream of person detections.
//
// The detector reports every person it sees, a few times per second, each with a box, a confidence and a small
// colour signature of their clothing. This class decides which of them is the presenter and where the presenter is
// right now:
//  - a new presenter is only accepted after being seen several times in a row (one-off false detections are ignored);
//  - once locked, a detection is the presenter only if it is where the presenter could have moved to in the time since
//    the last sighting, is about the same size, and wears the same colours;
//  - between detections, and when the presenter is briefly hidden, the position is predicted from the motion so far;
//  - if the presenter is lost, the camera holds. The same person is taken back at once when they reappear; anyone
//    else has to wait until the hold time has passed.
// No dependency on OBS or on the detector, so it can be tested on its own.

#include <string>
#include <vector>

#define PT_SIG 64 // 4x4x4 colour histogram

struct pt_det
{
	float x0, y0, x1, y1;
	float score;
	float sig[PT_SIG];
	bool has_sig;
};

class presenter_tracker {
public:
	enum state_e {
		st_search = 0, // nobody locked, looking for a presenter
		st_locked,     // following, confirmed by the latest detection
		st_coast,      // following on prediction, not seen for a moment
		st_lost,       // not seen for too long: holding
	};

	// settings
	float conf_new = 0.45f;  // confidence needed to become the presenter
	float conf_keep = 0.20f; // confidence that is enough to stay the presenter
	float lock_hold = 10.0f; // s. How long a lost presenter keeps the lock before somebody else may take over
	float coast_time = 1.0f; // s. How long to keep predicting without a detection
	int confirm_hits = 3;    // detections in a row before a new presenter is accepted

	presenter_tracker() { reset(); }
	void reset();

	// Feed one detector result. t = time the analysed picture was taken, in seconds.
	void update(const std::vector<pt_det> &dets, double t);

	// Where the presenter is at time t. Returns false when there is nobody to follow (camera should hold).
	bool predict(double t, float &x0, float &y0, float &x1, float &y1) const;

	// How fast the presenter is moving across the picture, pixels per second (0 when nobody is being followed).
	void velocity(float &vx, float &vy) const;

	state_e state() const { return st; }
	// One line for the log each time something worth knowing happened; empty otherwise.
	std::string take_event();

private:
	struct axis_s
	{
		float p, v;       // position, speed
		float pp, pv, vv; // covariance
	};
	static void ax_init(axis_s &a, float p, float r);
	static void ax_predict(axis_s &a, float dt, float q);
	static void ax_update(axis_s &a, float z, float r);

	state_e st;
	axis_s cx, top; // centre x and top edge of the box
	float w, h;     // box size
	double t_state; // time the filter state refers to
	double t_seen;  // time of the last confirming detection
	float sig[PT_SIG];
	bool have_sig;

	// candidate waiting to be confirmed
	bool cand_valid;
	float cand_cx, cand_top, cand_h;
	int cand_hits;
	double cand_t;

	std::string event;

	void lock_on(const pt_det &d, double t, const char *why);
	void advance(double t);
	static float similarity(const float *a, const float *b);
};
