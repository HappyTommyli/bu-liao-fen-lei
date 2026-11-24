#include <cassert>
#include <string>
#include "../firmware/mlx_c3_monitor/mlx_c3_monitor.ino"
int main(int argc,char**argv){
 assert(argc==2);std::string scenario=argv[1];
 if(scenario=="known_pins"){
  setup();assert(activeSda==3&&activeScl==4&&mlxAddr==0x0C);
  assert(Wire.beginCount==1);return 0;
 }
 if(scenario=="stream"){
  setup();assert(activeSda==3&&activeScl==4&&mlxAddr==0x0C);
  assert(Serial.output.str().find("READY\n")!=std::string::npos);
  Serial.output.str("");fakeUs+=3000;loop();
  assert(Serial.output.str().find(",100.000,20.000,30.000\n")!=std::string::npos);
  return 0;
 }
 if(scenario=="invalid"){
  Serial.inject("reboot\nREBOOT X\nXXXXXXXXXXXXXXXXXXXXXXXXREBOOT\n");
  try{for(int i=0;i<8;++i)loop();}catch(Restarted&){return 2;}
  assert(Serial.output.str().find("reboot requested")==std::string::npos);return 0;
 }
 if(scenario=="partial"){
  Serial.inject("REB");loop();assert(Serial.flushes==0);Serial.inject("OOT\r\n");
 }else Serial.inject("REBOOT\n");
 try{
  if(scenario=="warmup") {fakeReadFailure=true;warmupAndTare();}
  else loop();
 }catch(Restarted&){
  const std::string out=Serial.output.str();
  assert(out.find("# fw=mlx_c3 reboot requested\n")!=std::string::npos);
  assert(Serial.flushes==1);assert(fakeUs>=200000);return 0;
 }
 return 1;
}
