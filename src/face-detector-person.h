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
	int last_ms();
};
