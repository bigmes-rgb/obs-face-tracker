#pragma once
#include "face-detector-base.h"
#include "presenter-tracker.hpp"

// Finds people (whole bodies) with a small neural network (YOLOX, run by ncnn).
// A body is many times larger in the picture than a face and looks the same from the front, the side and the back,
// so the presenter is not lost when turning away from the camera.
class face_detector_person : public face_detector_base {
	struct face_detector_person_private_s *p;

	void detect_main() override;

public:
	face_detector_person();
	virtual ~face_detector_person();
	void set_texture(std::shared_ptr<texture_object> &, int crop_l, int crop_r, int crop_t, int crop_b) override;
	void get_faces(std::vector<struct rect_s> &) override; // the person boxes, for drawing
	void get_persons(std::vector<pt_det> &);
	void set_model(const char *param_file, const char *bin_file);
	// The body-pose model, and the box (full-size picture) of the person whose pose is wanted. on = false: no pose.
	void set_pose(const char *param_file, const char *bin_file, bool on, float x0, float y0, float x1, float y1);
	int last_pose_ms();
	void set_time(double t); // when the picture handed over with set_texture was taken, in seconds
	// How fast the background slid across the picture between the last two analysed pictures (px/s). This is the
	// camera's own movement. False when it could not be measured.
	bool get_background_speed(float &vx, float &vy);
	int last_ms();
};
