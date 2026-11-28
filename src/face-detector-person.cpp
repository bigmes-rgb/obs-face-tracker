// ncnn has to come before the OBS headers on Windows: it needs the complete Windows threading API, which a later
// include would otherwise narrow down.
#ifdef _WIN32
#undef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif
#include "person-detect-core.hpp"

#include <obs-module.h>
#include <util/platform.h>
#include <util/threading.h>
#include <algorithm>
#include <cmath>
#include <string>
#include "plugin-macros.generated.h"
#include "face-detector-person.h"
#include "texture-object.h"

#define MAX_ERROR 2

struct face_detector_person_private_s
{
	std::shared_ptr<texture_object> tex;
	std::vector<pt_det> persons;
	ncnn::Net net;
	bool net_loaded = false;
	bool has_error = false;
	std::string param_file, bin_file;
	int crop_l = 0, crop_r = 0, crop_t = 0, crop_b = 0;
	int n_error = 0;
	int last_ms = 0;
	pd_ego_state ego;
	double t_frame = 0.0, t_prev = 0.0; // when the picture being analysed, and the one before, were taken
	bool ego_ok = false;
	float ego_vx = 0.0f, ego_vy = 0.0f; // background movement, px/s of the full-size picture
};

face_detector_person::face_detector_person()
{
	p = new face_detector_person_private_s;
}

face_detector_person::~face_detector_person()
{
	delete p;
}

void face_detector_person::set_texture(std::shared_ptr<texture_object> &tex, int crop_l, int crop_r, int crop_t,
				       int crop_b)
{
	p->tex = tex;
	p->crop_l = crop_l;
	p->crop_r = crop_r;
	p->crop_t = crop_t;
	p->crop_b = crop_b;
}

void face_detector_person::set_model(const char *param_file, const char *bin_file)
{
	if (p->param_file != param_file || p->bin_file != bin_file) {
		p->param_file = param_file;
		p->bin_file = bin_file;
		p->net_loaded = false;
	}
}

void face_detector_person::set_time(double t)
{
	p->t_frame = t;
}

bool face_detector_person::get_background_speed(float &vx, float &vy)
{
	vx = p->ego_vx;
	vy = p->ego_vy;
	return p->ego_ok;
}

int face_detector_person::last_ms()
{
	return p->last_ms;
}

void face_detector_person::detect_main()
{
	if (!p->tex)
		return;

	const uint64_t t_start = os_gettime_ns();

	dlib::matrix<dlib::rgb_pixel> img;
	if (!p->tex->get_dlib_rgb_image(img)) {
		p->tex.reset();
		return;
	}
	const float scale = p->tex->scale;

	// the search area
	const int cx0 = std::max(0, (int)(p->crop_l / scale));
	const int cy0 = std::max(0, (int)(p->crop_t / scale));
	const int cx1 = (int)img.nc() - std::max(0, (int)(p->crop_r / scale));
	const int cy1 = (int)img.nr() - std::max(0, (int)(p->crop_b / scale));
	const int cw = cx1 - cx0, ch = cy1 - cy0;
	if (cw < 32 || ch < 32) {
		if (p->n_error++ < MAX_ERROR)
			blog(LOG_ERROR, "search area too small: %dx%d of a %dx%d picture", cw, ch, (int)img.nc(),
			     (int)img.nr());
		p->persons.clear();
		p->tex.reset();
		return;
	}

	if (!p->net_loaded) {
		p->net_loaded = true;
		p->net.clear();
		p->net.opt.use_vulkan_compute = false;
		p->net.opt.num_threads = 1;
		blog(LOG_INFO, "person detector: loading '%s'", p->param_file.c_str());
		p->has_error = p->net.load_param(p->param_file.c_str()) != 0 ||
			       p->net.load_model(p->bin_file.c_str()) != 0;
		if (p->has_error)
			blog(LOG_ERROR, "person detector: could not load the model files '%s' and '%s'",
			     p->param_file.c_str(), p->bin_file.c_str());
	}
	if (p->has_error) {
		p->persons.clear();
		p->tex.reset();
		return;
	}

	std::vector<pt_det> keep;
	bool ego_ok = false;
	float ego_dx = 0.0f, ego_dy = 0.0f;
	const bool ran =
		person_detect_core(p->net, img, cx0, cy0, cx1, cy1, scale, keep, &p->ego, &ego_ok, &ego_dx, &ego_dy);
	const float dt = (float)(p->t_frame - p->t_prev);
	p->ego_ok = ego_ok && dt > 0.02f && dt < 1.0f;
	if (p->ego_ok) {
		p->ego_vx = ego_dx / dt;
		p->ego_vy = ego_dy / dt;
	}
	p->t_prev = p->t_frame;
	if (!ran) {
		if (p->n_error++ < MAX_ERROR)
			blog(LOG_ERROR, "person detector: the network gave no result");
		p->persons.clear();
		p->tex.reset();
		return;
	}
	p->persons.swap(keep);
	if (p->n_error)
		p->n_error--;
	p->last_ms = (int)((os_gettime_ns() - t_start) / 1000000);
	p->tex.reset();
}

void face_detector_person::get_persons(std::vector<pt_det> &persons)
{
	persons = p->persons;
}

void face_detector_person::get_faces(std::vector<struct rect_s> &rects)
{
	rects.resize(p->persons.size());
	for (size_t i = 0; i < p->persons.size(); i++) {
		const pt_det &d = p->persons[i];
		rects[i] = rect_s{(int)d.x0, (int)d.y0, (int)d.x1, (int)d.y1, d.score};
	}
}
