#pragma once
#include "Wire.h"
enum mlx90393_gain_t{MLX90393_GAIN_1X};
enum mlx90393_resolution_t{MLX90393_RES_16};
enum mlx90393_oversampling_t{MLX90393_OSR_0};
enum mlx90393_filter_t{MLX90393_FILTER_0};
enum mlx90393_axis{MLX90393_X,MLX90393_Y,MLX90393_Z};
static bool fakeReadFailure=false;
class Adafruit_MLX90393{
 public:
 bool begin_I2C(uint8_t,FakeWire*){return true;}
 void setGain(mlx90393_gain_t){}void setResolution(mlx90393_axis,mlx90393_resolution_t){}
 void setOversampling(mlx90393_oversampling_t){}void setFilter(mlx90393_filter_t){}
 bool startSingleMeasurement(){return true;}
 bool readMeasurement(float*x,float*y,float*z){*x=100;*y=20;*z=30;return !fakeReadFailure;}
};
