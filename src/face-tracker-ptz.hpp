#pragma once

#include <vector>
#include <deque>
#include "helper.hpp"

struct face_tracker_ptz
{
	obs_source_t *context;
	uint32_t known_width;
	uint32_t known_height;
	bool rendered;
	uint64_t track_interval_ns; // minimum time between frames handed to the detector/trackers; 0 = every frame
	uint64_t last_stage_ns;
	uint32_t diag_width, diag_height;
	float diag_scale;
	bool is_active;

	video_scaler_t *scaler;
	uint8_t *scaler_buffer;
	struct video_scale_info scaler_src_info;
	struct video_scale_info scaler_dst_info;

	f3 detect_err;
	bool face_found, face_found_last;

	class ft_manager_for_ftptz *ftm;

	float track_z, track_x, track_y;

	float kp_x, kp_y, kp_z;
	f3 ki;
	f3 klpf;
	f3 tlpf;
	f3 e_deadband, e_nonlinear; // deadband and nonlinear amount for error input
	// Zone framing: the camera rests while the presenter is inside the rest zone around the target point. When
	// they leave it, the camera brings them back to the middle (inside the small settle zone) and rests again.
	bool zone_mode;
	float zone_x, zone_y;    // half size of the rest zone, as a fraction of the picture width / height
	bool zone_moving[2];     // pan, tilt: currently following
	float zone_anticipation; // 0 = follow after the person, 1 = move with the person
	float zone_cmd[2];       // what is being asked of the head, as an equivalent position error in pixels
	float zone_rate[2];      // how fast the camera itself is believed to be sweeping the picture, px/s
	float zone_v[2];         // the person's own speed (camera motion taken out), px/s
	float zone_fast_s[2];    // for how long the person has been walking
	float zone_cmd_lag[2];   // zone_cmd, delayed the way the head eases into a speed
	float zone_gain[2]; // how strongly the head really answers, compared with what the settings assume (learned)
	uint64_t zone_gain_log_ns;
	f3 filter_int;
	f3 filter_lpf;
	float f_att_int;
	int u[3];
	float u_linear[3];
	float ptz_query[3];
	uint64_t face_found_last_ns;
	int face_lost_preset_sent;

	int face_lost_preset_timeout_ms;
	int face_lost_ptz_preset;
	int face_lost_zoomout_timeout_ms;

	bool debug_faces;
	bool debug_notrack;
	bool debug_always_show;
	FILE *debug_data_tracker;
	FILE *debug_data_error;
	FILE *debug_data_control;
	char *debug_data_tracker_last;
	char *debug_data_error_last;
	char *debug_data_control_last;

	char *ptz_type;

	bool is_paused;
	obs_hotkey_pair_id hotkey_pause;
	obs_hotkey_id hotkey_reset;
	obs_hotkey_id hotkey_lock;
	long long lock_request_seen; // last value of the "lock_request" setting (changed by the phone page)
	float zone_strength;         // how firmly the person is brought back to the middle (1 = normal)
};
