#pragma once
#include <cstdint>
#include <cmath>
#include <deque>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
static uint64_t fakeUs=0;
inline uint32_t micros(){return static_cast<uint32_t>(fakeUs);}
inline uint32_t millis(){return static_cast<uint32_t>(fakeUs/1000);}
inline void delay(uint32_t ms){fakeUs+=ms*1000;}
const int HEX=16;
struct Restarted{};
struct FakeSerial{
 std::deque<char> input;std::ostringstream output;int flushes=0;
 void begin(uint32_t){} explicit operator bool()const{return true;}
 int available()const{return input.size();} int read(){char v=input.front();input.pop_front();return v;}
 void inject(const std::string&s){for(char c:s)input.push_back(c);}
 void flush(){++flushes;}
 template<class T>void print(T v){output<<v;}
 void print(uint8_t v){output<<static_cast<unsigned>(v);}
 void print(uint8_t v,int base){if(base==HEX)output<<std::hex;output<<static_cast<unsigned>(v)<<std::dec;}
 void print(float v,int digits){output<<std::fixed<<std::setprecision(digits)<<v;}
 void print(double v,int digits){output<<std::fixed<<std::setprecision(digits)<<v;}
 void println(){output<<'\n';}
 template<class T>void println(T v){print(v);println();}
 void println(uint8_t v,int base){print(v,base);println();}
 void println(float v,int digits){print(v,digits);println();}
};
static FakeSerial Serial;
struct FakeEsp{void restart(){throw Restarted{};}};static FakeEsp ESP __attribute__((unused));
struct FakeWire{
 uint8_t sda=0,scl=0,address=0;unsigned beginCount=0;
 void end(){}void begin(uint8_t a,uint8_t b){sda=a;scl=b;++beginCount;}
 void setClock(uint32_t){}void setTimeOut(uint16_t){}
 void beginTransmission(uint8_t a){address=a;}
 uint8_t endTransmission(){return sda==3&&scl==4&&address==0x0C?0:2;}
};static FakeWire Wire;
