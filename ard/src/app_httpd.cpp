
#include "esp_http_server.h"
#include "esp_timer.h"
#include "esp_camera.h"
#include "img_converters.h"
#include "camera_index.h"
#include "Arduino.h"




typedef struct
{
  httpd_req_t *req;
  size_t len;
} jpg_chunking_t;



#define PART_BOUNDARY "123456789000000000000987654321"
static const char *_STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char *_STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char *_STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";



httpd_handle_t stream_httpd = NULL;
httpd_handle_t camera_httpd = NULL;


static int8_t detection_enabled = 0;
static int8_t recognition_enabled = 0;
static int8_t is_enrolling = 0;


static uint8_t iterations = 2;

static uint8_t Q = 70;
static float dt = 0.1;

static float z = -0.5*dt;
static float A[9] = {dt*0, dt*0, dt*0, dt*0, dt*2, dt*0, dt*0, dt*0, dt*0};
static float B[9] = {-1*dt, -1*dt, -1*dt, -1*dt, 8*dt, -1*dt, -1*dt, -1*dt, -1*dt};

static float zz = -3;
static float AA[9] = {0, 0, 0, 0, 2, 0, 0, 0, 0};
static float BB[9] = {-1, -1, -1, -1, 8, -1, -1, -1, -1};

float pc[256];



typedef struct {
  camera_fb_t *fb;
  uint8_t iterations;
} params_t;

void fill_pc_array() {
  for (int i = 0; i < 256; i++) {
    pc[i] = 1 - 2 * (i / 255.0);
  }
}




static size_t jpg_encode_stream(void *arg, size_t index, const void *data, size_t len)
{ // només se crida a capture handler
  jpg_chunking_t *j = (jpg_chunking_t *)arg;
  if (!index)
  {
    j->len = 0;
  }
  if (httpd_resp_send_chunk(j->req, (const char *)data, len) != ESP_OK)
  {
    return 0;
  }
  j->len += len;
  return len;
}


// Semaphores
SemaphoreHandle_t semaphore1=xSemaphoreCreateBinary();
SemaphoreHandle_t semaphore2=xSemaphoreCreateBinary();

// CNN left side of fb
void CNN_left(void *pvParameters){
  params_t *params = (params_t *)pvParameters;
  camera_fb_t *fb = params->fb;
  uint8_t iterations=params->iterations;
  int w=fb->width;
  int h=fb->height;
  int len=fb->len;
  float a=1-dt;
  float div=-255.0/2;
  int initial=0; //0:0, 1:1,2:U
  const int iter=iterations;
  float aa=a+  A[0]  + A[1] + A[2]  + A[3]  + A[4]  + A[5]  + A[6]  + A[7]  + A[8];

  int buffer_index = 0;
  int buffer_index_x = 0;
  int mid=h;
  
  // Circular queues
  float cby[iter][3][w/2];
  float cbx[iter][w/2];
  float cbs[iter][w/2];

  uint8_t buf8[(w-1)/2];

    //Pointers to circular queues
    float* y[iter][3];
    float* x[iter];
    float* s[iter];
    float* r;   
    float* xp;
    float* sp;

  for (int i = 0; i < h-1+iter; i++) {

    // Update pointers
    buffer_index=(buffer_index + 1) % 3;
    buffer_index_x=(buffer_index_x + 1) % (iter);
    for(int j=0;j<iter;j++){
        x[j]=cbx[(buffer_index_x + (iter-1-j)) % (iter)];
        s[j]=cbs[(buffer_index_x + (iter-1-j)) % (iter)];
        for (int k = 0; k <3; k++){
          y[j][k]=cby[j][(buffer_index + k) % 3];
        }
    }
    if(i<h-1){

      // Fill Y circular queue. it=0
    uint8_t* ptr = fb->buf + (i+1) * w ;
    for (int j = 1; j < w/2; j+=4) { 
        y[0][2][j]=pc[*(++ptr)];
        y[0][2][j+1]=pc[*(++ptr)];
        y[0][2][j+2]=pc[*(++ptr)];
        y[0][2][j+3]=pc[*(++ptr)];
    } 


    r = y[1][2];  
    xp = x[0];
    sp = s[0];

    // First iteration
    if(initial==0){
    
    for (int j = 1; j < w/2; j+=4){
      float x1 = B[0] * y[0][0][j-1] + B[1] * y[0][0][j] + B[2] * y[0][0][j+1] + B[3] * y[0][1][j-1] + B[4] * y[0][1][j] + B[5] * y[0][1][j+1] + B[6] * y[0][2][j-1] + B[7] * y[0][2][j] + B[8] * y[0][2][j+1]+z;
      float x2 = B[0] * y[0][0][j] + B[1] * y[0][0][j+1] + B[2] * y[0][0][j+2] + B[3] * y[0][1][j] + B[4] * y[0][1][j+1] + B[5] * y[0][1][j+2] + B[6] * y[0][2][j] + B[7] * y[0][2][j+1] + B[8] * y[0][2][j+2]+z;
      float x3 = B[0] * y[0][0][j+1] + B[1] * y[0][0][j+2] + B[2] * y[0][0][j+3] + B[3] * y[0][1][j+1] + B[4] * y[0][1][j+2] + B[5] * y[0][1][j+3] + B[6] * y[0][2][j+1] + B[7] * y[0][2][j+2] + B[8] * y[0][2][j+3]+z;
      float x4 = B[0] * y[0][0][j+2] + B[1] * y[0][0][j+3] + B[2] * y[0][0][j+4] + B[3] * y[0][1][j+2] + B[4] * y[0][1][j+3] + B[5] * y[0][1][j+4] + B[6] * y[0][2][j+2] + B[7] * y[0][2][j+3] + B[8] * y[0][2][j+4]+z;
      *(++xp)=*(++sp)=*(++r)=x1;
      *(++xp)=*(++sp)=*(++r)=x2;
      *(++xp)=*(++sp)=*(++r)=x3;
      *(++xp)=*(++sp)=*(++r)=x4;
            if(fabs(*r)>1){
              if(*r<0){*r=-1;}
              else if(*r>0){*r=1;}
            }
            if(fabs(*(r-1))>1){
              if(*(r-1)<0){*(r-1)=-1;}
              else if(*(r-1)>0){*(r-1)=1;}
            }
            if(fabs(*(r-2))>1){
              if(*(r-2)<0){*(r-2)=-1;}
              else if(*(r-2)>0){*(r-2)=1;}
            }
            if(fabs(*(r-3))>1){
              if(*(r-3)<0){*(r-3)=-1;}
              else if(*(r-3)>0){*(r-3)=1;}
            }
    }
    }
    else if(initial==1){
      
        for (int j = 1; j < w/2; j+=4){
      float x1 = B[0] * y[0][0][j-1] + B[1] * y[0][0][j] + B[2] * y[0][0][j+1] + B[3] * y[0][1][j-1] + B[4] * y[0][1][j] + B[5] * y[0][1][j+1] + B[6] * y[0][2][j-1] + B[7] * y[0][2][j] + B[8] * y[0][2][j+1]+z;
      float x2 = B[0] * y[0][0][j] + B[1] * y[0][0][j+1] + B[2] * y[0][0][j+2] + B[3] * y[0][1][j] + B[4] * y[0][1][j+1] + B[5] * y[0][1][j+2] + B[6] * y[0][2][j] + B[7] * y[0][2][j+1] + B[8] * y[0][2][j+2]+z;
      float x3 = B[0] * y[0][0][j+1] + B[1] * y[0][0][j+2] + B[2] * y[0][0][j+3] + B[3] * y[0][1][j+1] + B[4] * y[0][1][j+2] + B[5] * y[0][1][j+3] + B[6] * y[0][2][j+1] + B[7] * y[0][2][j+2] + B[8] * y[0][2][j+3]+z;
      float x4 = B[0] * y[0][0][j+2] + B[1] * y[0][0][j+3] + B[2] * y[0][0][j+4] + B[3] * y[0][1][j+2] + B[4] * y[0][1][j+3] + B[5] * y[0][1][j+4] + B[6] * y[0][2][j+2] + B[7] * y[0][2][j+3] + B[8] * y[0][2][j+4]+z;
      *(++sp)=x1;
      *(++sp)=x2;
      *(++sp)=x3;
      *(++sp)=x4;

      x1 += aa;
      x2 += aa;
      x3 += aa;
      x4 += aa;           
           *(++xp)=*(++r)=x1;
           *(++xp)=*(++r)=x2;
           *(++xp)=*(++r)=x3;
           *(++xp)=*(++r)=x4;
            if(fabs(*r)>1){
              if(*r<0){*r=-1;}
              else if(*r>0){*r=1;}
            }
            if(fabs(*(r-1))>1){
              if(*(r-1)<0){*(r-1)=-1;}
              else if(*(r-1)>0){*(r-1)=1;}
            }
            if(fabs(*(r-2))>1){
              if(*(r-2)<0){*(r-2)=-1;}
              else if(*(r-2)>0){*(r-2)=1;}
            }
            if(fabs(*(r-3))>1){
              if(*(r-3)<0){*(r-3)=-1;}
              else if(*(r-3)>0){*(r-3)=1;}
            }
    }
    }
    else{
          for (int j = 1; j < w/2; j+=4){
      float x1 = B[0] * y[0][0][j-1] + B[1] * y[0][0][j] + B[2] * y[0][0][j+1] + B[3] * y[0][1][j-1] + B[4] * y[0][1][j] + B[5] * y[0][1][j+1] + B[6] * y[0][2][j-1] + B[7] * y[0][2][j] + B[8] * y[0][2][j+1]+z;
      float x2 = B[0] * y[0][0][j] + B[1] * y[0][0][j+1] + B[2] * y[0][0][j+2] + B[3] * y[0][1][j] + B[4] * y[0][1][j+1] + B[5] * y[0][1][j+2] + B[6] * y[0][2][j] + B[7] * y[0][2][j+1] + B[8] * y[0][2][j+2]+z;
      float x3 = B[0] * y[0][0][j+1] + B[1] * y[0][0][j+2] + B[2] * y[0][0][j+3] + B[3] * y[0][1][j+1] + B[4] * y[0][1][j+2] + B[5] * y[0][1][j+3] + B[6] * y[0][2][j+1] + B[7] * y[0][2][j+2] + B[8] * y[0][2][j+3]+z;
      float x4 = B[0] * y[0][0][j+2] + B[1] * y[0][0][j+3] + B[2] * y[0][0][j+4] + B[3] * y[0][1][j+2] + B[4] * y[0][1][j+3] + B[5] * y[0][1][j+4] + B[6] * y[0][2][j+2] + B[7] * y[0][2][j+3] + B[8] * y[0][2][j+4]+z;
      *(++sp)=x1;
      *(++sp)=x2;
      *(++sp)=x3;
      *(++sp)=x4;
      x1 += a*y[0][1][j] +  A[0] * y[0][0][j-1] + A[1] * y[0][0][j] + A[2] * y[0][0][j+1] + A[3] * y[0][1][j-1] + A[4] * y[0][1][j] + A[5] * y[0][1][j+1] + A[6] * y[0][2][j-1] + A[7] * y[0][2][j] + A[8] * y[0][2][j+1];
      x2 += a*y[0][1][j+1] + A[0] * y[0][0][j] + A[1] * y[0][0][j+1] + A[2] * y[0][0][j+2] + A[3] * y[0][1][j] + A[4] * y[0][1][j+1] + A[5] * y[0][1][j+2] + A[6] * y[0][2][j] + A[7] * y[0][2][j+1] + A[8] * y[0][2][j+2];
      x3 += a*y[0][1][j+2] +A[0] * y[0][0][j+1] + A[1] * y[0][0][j+2] + A[2] * y[0][0][j+3] + A[3] * y[0][1][j+1] + A[4] * y[0][1][j+2] + A[5] * y[0][1][j+3] + A[6] * y[0][2][j+1] + A[7] * y[0][2][j+2] + A[8] * y[0][2][j+3];
      x4 += a*y[0][1][j+3] +A[0] * y[0][0][j+2] + A[1] * y[0][0][j+3] + A[2] * y[0][0][j+4] + A[3] * y[0][1][j+2] + A[4] * y[0][1][j+3] + A[5] * y[0][1][j+4] + A[6] * y[0][2][j+2] + A[7] * y[0][2][j+3] + A[8] * y[0][2][j+4];           
           *(++xp)=*(++r)=x1;
           *(++xp)=*(++r)=x2;
           *(++xp)=*(++r)=x3;
           *(++xp)=*(++r)=x4;
            if(fabs(*r)>1){
              if(*r<0){*r=-1;}
              else if(*r>0){*r=1;}
            }
            if(fabs(*(r-1))>1){
              if(*(r-1)<0){*(r-1)=-1;}
              else if(*(r-1)>0){*(r-1)=1;}
            }
            if(fabs(*(r-2))>1){
              if(*(r-2)<0){*(r-2)=-1;}
              else if(*(r-2)>0){*(r-2)=1;}
            }
            if(fabs(*(r-3))>1){
              if(*(r-3)<0){*(r-3)=-1;}
              else if(*(r-3)>0){*(r-3)=1;}
            }
    }
    }

    }
    
    // Intermediate iterations
    for (int it = 1; it < iter-1; it++)
    {     
        if(i>it){
          r = y[it+1][2];
          xp = x[it]; 
          sp = s[it];
          for (int j = 1; j < w/2; j+=4){ 
            float x1 = *(++xp)*a + *(++sp) + A[0] * y[it][0][j-1] + A[1] * y[it][0][j] + A[2] * y[it][0][j+1] + A[3] * y[it][1][j-1] + A[4] * y[it][1][j] + A[5] * y[it][1][j+1] + A[6] * y[it][2][j-1] + A[7] * y[it][2][j] + A[8] * y[it][2][j+1];
            float x2 = *(++xp)*a + *(++sp) + A[0] * y[it][0][j] + A[1] * y[it][0][j+1] + A[2] * y[it][0][j+2] + A[3] * y[it][1][j] + A[4] * y[it][1][j+1] + A[5] * y[it][1][j+2] + A[6] * y[it][2][j] + A[7] * y[it][2][j+1] + A[8] * y[it][2][j+2];
            float x3 = *(++xp)*a + *(++sp) + A[0] * y[it][0][j+1] + A[1] * y[it][0][j+2] + A[2] * y[it][0][j+3] + A[3] * y[it][1][j+1] + A[4] * y[it][1][j+2] + A[5] * y[it][1][j+3] + A[6] * y[it][2][j+1] + A[7] * y[it][2][j+2] + A[8] * y[it][2][j+3];
            float x4 = *(++xp)*a + *(++sp) + A[0] * y[it][0][j+2] + A[1] * y[it][0][j+3] + A[2] * y[it][0][j+4] + A[3] * y[it][1][j+2] + A[4] * y[it][1][j+3] + A[5] * y[it][1][j+4] + A[6] * y[it][2][j+2] + A[7] * y[it][2][j+3] + A[8] * y[it][2][j+4];
            *(xp-3)=*(++r)=x1;
            *(xp-2)=*(++r)=x2;
            *(xp-1)=*(++r)=x3;
            *(xp)=*(++r)=x4;

            if(fabs(*r)>1){
              if(*r<0){*r=-1;}
              else if(*r>0){*r=1;}
            }
            if(fabs(*(r-1))>1){
              if(*(r-1)<0){*(r-1)=-1;}
              else if(*(r-1)>0){*(r-1)=1;}
            }
            if(fabs(*(r-2))>1){
              if(*(r-2)<0){*(r-2)=-1;}
              else if(*(r-2)>0){*(r-2)=1;}
            }
            if(fabs(*(r-3))>1){
              if(*(r-3)<0){*(r-3)=-1;}
              else if(*(r-3)>0){*(r-3)=1;}
            }
          
          }


    }

    }
    

    //Final iteration
    if(i>iter){
      uint8_t* bufp = buf8-1;
      xp = x[iter-1]; 
      sp = s[iter-1];
      for (int j = 1; j < w/2; j+=4){ 
        float x1 = *(++xp) * a + *(++sp)  + A[0] * y[iter-1][0][j-1] + A[1] * y[iter-1][0][j] + A[2] * y[iter-1][0][j+1] + A[3] * y[iter-1][1][j-1] + A[4] * y[iter-1][1][j] + A[5] * y[iter-1][1][j+1] + A[6] * y[iter-1][2][j-1] + A[7] * y[iter-1][2][j] + A[8] * y[iter-1][2][j+1];
        float x2 = *(++xp) * a + *(++sp)  + A[0] * y[iter-1][0][j] + A[1] * y[iter-1][0][j+1] + A[2] * y[iter-1][0][j+2] + A[3] * y[iter-1][1][j] + A[4] * y[iter-1][1][j+1] + A[5] * y[iter-1][1][j+2] + A[6] * y[iter-1][2][j] + A[7] * y[iter-1][2][j+1] + A[8] * y[iter-1][2][j+2];
        float x3 = *(++xp) * a + *(++sp)  + A[0] * y[iter-1][0][j+1] + A[1] * y[iter-1][0][j+2] + A[2] * y[iter-1][0][j+3] + A[3] * y[iter-1][1][j+1] + A[4] * y[iter-1][1][j+2] + A[5] * y[iter-1][1][j+3] + A[6] * y[iter-1][2][j+1] + A[7] * y[iter-1][2][j+2] + A[8] * y[iter-1][2][j+3];
        float x4 = *(++xp) * a + *(++sp)  + A[0] * y[iter-1][0][j+2] + A[1] * y[iter-1][0][j+3] + A[2] * y[iter-1][0][j+4] + A[3] * y[iter-1][1][j+2] + A[4] * y[iter-1][1][j+3] + A[5] * y[iter-1][1][j+4] + A[6] * y[iter-1][2][j+2] + A[7] * y[iter-1][2][j+3] + A[8] * y[iter-1][2][j+4];
        if(fabs(x1)>1){
          if(x1<0){*(++bufp)=255;}
          else if(x1>0){*(++bufp)=0;}
        }
        else{*(++bufp)=(x1-1)*div;}
        if(fabs(x2)>1){
          if(x2<0){*(++bufp)=255;}
          else if(x2>0){*(++bufp)=0;}
        }
        else{*(++bufp)=(x2-1)*div;}
        if(fabs(x3)>1){
          if(x3<0){*(++bufp)=255;}
          else if(x3>0){*(++bufp)=0;}
        }
        else{*(++bufp)=(x3-1)*div;}
        if(fabs(x4)>1){
          if(x4<0){*(++bufp)=255;}
          else if(x4>0){*(++bufp)=0;}
        }
        else{*(++bufp)=(x4-1)*div;}
      }
      memcpy(&fb->buf[(i-iter)*w+1],buf8,((w-1)/2-6) * sizeof(uint8_t));
    }

  
  }

      xSemaphoreGive(semaphore1);
          vTaskDelete(NULL); 
}

// CNN right side of fb
void CNN_right(void *pvParameters){
      params_t *params = (params_t *)pvParameters;
  camera_fb_t *fb = params->fb;
  uint8_t iterations=params->iterations;
  int w=fb->width;
  int h=fb->height;
  int len=fb->len;
  //dt = 0.001;
  float a=1-dt;
  float div=-255.0/2;
  int initial=0; //0:0, 1:1,2:U
float aa=a+  A[0]  + A[1] + A[2]  + A[3]  + A[4]  + A[5]  + A[6]  + A[7]  + A[8];
  const int iter=iterations;

  int buffer_index = 0;
  int buffer_index_x = 0; 
  int mid=h;
  int patch=8;
  

  float cby[iter][3][w/2+patch];
  float cbx[iter][w/2+patch];
  float cbs[iter][w/2+patch];
  uint8_t buf8[(w-1)/2+patch];

    float* y[iter][3];
    float* x[iter];
    float* s[iter];
    float* r ;   
    float* xp;
    float* sp;

  for (int i = 0; i < h-1+iter; i++) {
    buffer_index=(buffer_index + 1) % 3;
    buffer_index_x=(buffer_index_x + 1) % (iter);
    for(int j=0;j<iter;j++){
        x[j]=cbx[(buffer_index_x + (iter-1-j)) % (iter)];
        s[j]=cbs[(buffer_index_x + (iter-1-j)) % (iter)];
        for (int k = 0; k <3; k++){
          y[j][k]=cby[j][(buffer_index + k) % 3];
        }
    }
    if(i<h-1){
    uint8_t* ptr = fb->buf + (i+1) * w +w/2+1-patch;
    for (int j = 1; j < w/2+patch; j+=4) { 
        y[0][2][j]=pc[*(++ptr)];
        y[0][2][j+1]=pc[*(++ptr)];
        y[0][2][j+2]=pc[*(++ptr)];
        y[0][2][j+3]=pc[*(++ptr)];
    } 


    r = y[1][2];     
    xp = x[0];
    sp = s[0];
    
    if(initial==0){
    
    for (int j = 1; j < w/2; j+=4){
      float x1 = B[0] * y[0][0][j-1] + B[1] * y[0][0][j] + B[2] * y[0][0][j+1] + B[3] * y[0][1][j-1] + B[4] * y[0][1][j] + B[5] * y[0][1][j+1] + B[6] * y[0][2][j-1] + B[7] * y[0][2][j] + B[8] * y[0][2][j+1]+z;
      float x2 = B[0] * y[0][0][j] + B[1] * y[0][0][j+1] + B[2] * y[0][0][j+2] + B[3] * y[0][1][j] + B[4] * y[0][1][j+1] + B[5] * y[0][1][j+2] + B[6] * y[0][2][j] + B[7] * y[0][2][j+1] + B[8] * y[0][2][j+2]+z;
      float x3 = B[0] * y[0][0][j+1] + B[1] * y[0][0][j+2] + B[2] * y[0][0][j+3] + B[3] * y[0][1][j+1] + B[4] * y[0][1][j+2] + B[5] * y[0][1][j+3] + B[6] * y[0][2][j+1] + B[7] * y[0][2][j+2] + B[8] * y[0][2][j+3]+z;
      float x4 = B[0] * y[0][0][j+2] + B[1] * y[0][0][j+3] + B[2] * y[0][0][j+4] + B[3] * y[0][1][j+2] + B[4] * y[0][1][j+3] + B[5] * y[0][1][j+4] + B[6] * y[0][2][j+2] + B[7] * y[0][2][j+3] + B[8] * y[0][2][j+4]+z;
      *(++xp)=*(++sp)=*(++r)=x1;
      *(++xp)=*(++sp)=*(++r)=x2;
      *(++xp)=*(++sp)=*(++r)=x3;
      *(++xp)=*(++sp)=*(++r)=x4;
            if(fabs(*r)>1){
              if(*r<0){*r=-1;}
              else if(*r>0){*r=1;}
            }
            if(fabs(*(r-1))>1){
              if(*(r-1)<0){*(r-1)=-1;}
              else if(*(r-1)>0){*(r-1)=1;}
            }
            if(fabs(*(r-2))>1){
              if(*(r-2)<0){*(r-2)=-1;}
              else if(*(r-2)>0){*(r-2)=1;}
            }
            if(fabs(*(r-3))>1){
              if(*(r-3)<0){*(r-3)=-1;}
              else if(*(r-3)>0){*(r-3)=1;}
            }
    }
    }
        else if(initial==1){
      
        for (int j = 1; j < w/2; j+=4){
      float x1 = B[0] * y[0][0][j-1] + B[1] * y[0][0][j] + B[2] * y[0][0][j+1] + B[3] * y[0][1][j-1] + B[4] * y[0][1][j] + B[5] * y[0][1][j+1] + B[6] * y[0][2][j-1] + B[7] * y[0][2][j] + B[8] * y[0][2][j+1]+z;
      float x2 = B[0] * y[0][0][j] + B[1] * y[0][0][j+1] + B[2] * y[0][0][j+2] + B[3] * y[0][1][j] + B[4] * y[0][1][j+1] + B[5] * y[0][1][j+2] + B[6] * y[0][2][j] + B[7] * y[0][2][j+1] + B[8] * y[0][2][j+2]+z;
      float x3 = B[0] * y[0][0][j+1] + B[1] * y[0][0][j+2] + B[2] * y[0][0][j+3] + B[3] * y[0][1][j+1] + B[4] * y[0][1][j+2] + B[5] * y[0][1][j+3] + B[6] * y[0][2][j+1] + B[7] * y[0][2][j+2] + B[8] * y[0][2][j+3]+z;
      float x4 = B[0] * y[0][0][j+2] + B[1] * y[0][0][j+3] + B[2] * y[0][0][j+4] + B[3] * y[0][1][j+2] + B[4] * y[0][1][j+3] + B[5] * y[0][1][j+4] + B[6] * y[0][2][j+2] + B[7] * y[0][2][j+3] + B[8] * y[0][2][j+4]+z;
      *(++sp)=x1;
      *(++sp)=x2;
      *(++sp)=x3;
      *(++sp)=x4;

      x1 += aa;
      x2 += aa;
      x3 += aa;
      x4 += aa;           
           *(++xp)=*(++r)=x1;
           *(++xp)=*(++r)=x2;
           *(++xp)=*(++r)=x3;
           *(++xp)=*(++r)=x4;
            if(fabs(*r)>1){
              if(*r<0){*r=-1;}
              else if(*r>0){*r=1;}
            }
            if(fabs(*(r-1))>1){
              if(*(r-1)<0){*(r-1)=-1;}
              else if(*(r-1)>0){*(r-1)=1;}
            }
            if(fabs(*(r-2))>1){
              if(*(r-2)<0){*(r-2)=-1;}
              else if(*(r-2)>0){*(r-2)=1;}
            }
            if(fabs(*(r-3))>1){
              if(*(r-3)<0){*(r-3)=-1;}
              else if(*(r-3)>0){*(r-3)=1;}
            }
    }
    }
    
    else{
          for (int j = 1; j < w/2; j+=4){
      float x1 = B[0] * y[0][0][j-1] + B[1] * y[0][0][j] + B[2] * y[0][0][j+1] + B[3] * y[0][1][j-1] + B[4] * y[0][1][j] + B[5] * y[0][1][j+1] + B[6] * y[0][2][j-1] + B[7] * y[0][2][j] + B[8] * y[0][2][j+1]+z;
      float x2 = B[0] * y[0][0][j] + B[1] * y[0][0][j+1] + B[2] * y[0][0][j+2] + B[3] * y[0][1][j] + B[4] * y[0][1][j+1] + B[5] * y[0][1][j+2] + B[6] * y[0][2][j] + B[7] * y[0][2][j+1] + B[8] * y[0][2][j+2]+z;
      float x3 = B[0] * y[0][0][j+1] + B[1] * y[0][0][j+2] + B[2] * y[0][0][j+3] + B[3] * y[0][1][j+1] + B[4] * y[0][1][j+2] + B[5] * y[0][1][j+3] + B[6] * y[0][2][j+1] + B[7] * y[0][2][j+2] + B[8] * y[0][2][j+3]+z;
      float x4 = B[0] * y[0][0][j+2] + B[1] * y[0][0][j+3] + B[2] * y[0][0][j+4] + B[3] * y[0][1][j+2] + B[4] * y[0][1][j+3] + B[5] * y[0][1][j+4] + B[6] * y[0][2][j+2] + B[7] * y[0][2][j+3] + B[8] * y[0][2][j+4]+z;
      *(++sp)=x1;
      *(++sp)=x2;
      *(++sp)=x3;
      *(++sp)=x4;
      x1 += a*y[0][1][j] +  A[0] * y[0][0][j-1] + A[1] * y[0][0][j] + A[2] * y[0][0][j+1] + A[3] * y[0][1][j-1] + A[4] * y[0][1][j] + A[5] * y[0][1][j+1] + A[6] * y[0][2][j-1] + A[7] * y[0][2][j] + A[8] * y[0][2][j+1];
      x2 += a*y[0][1][j+1] + A[0] * y[0][0][j] + A[1] * y[0][0][j+1] + A[2] * y[0][0][j+2] + A[3] * y[0][1][j] + A[4] * y[0][1][j+1] + A[5] * y[0][1][j+2] + A[6] * y[0][2][j] + A[7] * y[0][2][j+1] + A[8] * y[0][2][j+2];
      x3 += a*y[0][1][j+2] +A[0] * y[0][0][j+1] + A[1] * y[0][0][j+2] + A[2] * y[0][0][j+3] + A[3] * y[0][1][j+1] + A[4] * y[0][1][j+2] + A[5] * y[0][1][j+3] + A[6] * y[0][2][j+1] + A[7] * y[0][2][j+2] + A[8] * y[0][2][j+3];
      x4 += a*y[0][1][j+3] +A[0] * y[0][0][j+2] + A[1] * y[0][0][j+3] + A[2] * y[0][0][j+4] + A[3] * y[0][1][j+2] + A[4] * y[0][1][j+3] + A[5] * y[0][1][j+4] + A[6] * y[0][2][j+2] + A[7] * y[0][2][j+3] + A[8] * y[0][2][j+4];           
           *(++xp)=*(++r)=x1;
           *(++xp)=*(++r)=x2;
           *(++xp)=*(++r)=x3;
           *(++xp)=*(++r)=x4;
            if(fabs(*r)>1){
              if(*r<0){*r=-1;}
              else if(*r>0){*r=1;}
            }
            if(fabs(*(r-1))>1){
              if(*(r-1)<0){*(r-1)=-1;}
              else if(*(r-1)>0){*(r-1)=1;}
            }
            if(fabs(*(r-2))>1){
              if(*(r-2)<0){*(r-2)=-1;}
              else if(*(r-2)>0){*(r-2)=1;}
            }
            if(fabs(*(r-3))>1){
              if(*(r-3)<0){*(r-3)=-1;}
              else if(*(r-3)>0){*(r-3)=1;}
            }
    }
    }

    }
    
    for (int it = 1; it < iter-1; it++)
    {     
        if(i>it){
          r = y[it+1][2];
          xp = x[it]; 
          sp = s[it];
          for (int j = 1; j < w/2; j+=4){ 
            float x1 = *(++xp)*a + *(++sp) + A[0] * y[it][0][j-1] + A[1] * y[it][0][j] + A[2] * y[it][0][j+1] + A[3] * y[it][1][j-1] + A[4] * y[it][1][j] + A[5] * y[it][1][j+1] + A[6] * y[it][2][j-1] + A[7] * y[it][2][j] + A[8] * y[it][2][j+1];
            float x2 = *(++xp)*a + *(++sp) + A[0] * y[it][0][j] + A[1] * y[it][0][j+1] + A[2] * y[it][0][j+2] + A[3] * y[it][1][j] + A[4] * y[it][1][j+1] + A[5] * y[it][1][j+2] + A[6] * y[it][2][j] + A[7] * y[it][2][j+1] + A[8] * y[it][2][j+2];
            float x3 = *(++xp)*a + *(++sp) + A[0] * y[it][0][j+1] + A[1] * y[it][0][j+2] + A[2] * y[it][0][j+3] + A[3] * y[it][1][j+1] + A[4] * y[it][1][j+2] + A[5] * y[it][1][j+3] + A[6] * y[it][2][j+1] + A[7] * y[it][2][j+2] + A[8] * y[it][2][j+3];
            float x4 = *(++xp)*a + *(++sp) + A[0] * y[it][0][j+2] + A[1] * y[it][0][j+3] + A[2] * y[it][0][j+4] + A[3] * y[it][1][j+2] + A[4] * y[it][1][j+3] + A[5] * y[it][1][j+4] + A[6] * y[it][2][j+2] + A[7] * y[it][2][j+3] + A[8] * y[it][2][j+4];
            *(xp-3)=*(++r)=x1;
            *(xp-2)=*(++r)=x2;
            *(xp-1)=*(++r)=x3;
            *(xp)=*(++r)=x4;

            if(fabs(*r)>1){
              if(*r<0){*r=-1;}
              else if(*r>0){*r=1;}
            }
            if(fabs(*(r-1))>1){
              if(*(r-1)<0){*(r-1)=-1;}
              else if(*(r-1)>0){*(r-1)=1;}
            }
            if(fabs(*(r-2))>1){
              if(*(r-2)<0){*(r-2)=-1;}
              else if(*(r-2)>0){*(r-2)=1;}
            }
            if(fabs(*(r-3))>1){
              if(*(r-3)<0){*(r-3)=-1;}
              else if(*(r-3)>0){*(r-3)=1;}
            }
          
          }


    }

    }
    

    if(i>iter){
      uint8_t* bufp = buf8-1;
      xp = x[iter-1]; 
      sp = s[iter-1];
      for (int j = 1; j < w/2; j+=4){ 
        float x1 = *(++xp) * a + *(++sp)  + A[0] * y[iter-1][0][j-1] + A[1] * y[iter-1][0][j] + A[2] * y[iter-1][0][j+1] + A[3] * y[iter-1][1][j-1] + A[4] * y[iter-1][1][j] + A[5] * y[iter-1][1][j+1] + A[6] * y[iter-1][2][j-1] + A[7] * y[iter-1][2][j] + A[8] * y[iter-1][2][j+1];
        float x2 = *(++xp) * a + *(++sp)  + A[0] * y[iter-1][0][j] + A[1] * y[iter-1][0][j+1] + A[2] * y[iter-1][0][j+2] + A[3] * y[iter-1][1][j] + A[4] * y[iter-1][1][j+1] + A[5] * y[iter-1][1][j+2] + A[6] * y[iter-1][2][j] + A[7] * y[iter-1][2][j+1] + A[8] * y[iter-1][2][j+2];
        float x3 = *(++xp) * a + *(++sp)  + A[0] * y[iter-1][0][j+1] + A[1] * y[iter-1][0][j+2] + A[2] * y[iter-1][0][j+3] + A[3] * y[iter-1][1][j+1] + A[4] * y[iter-1][1][j+2] + A[5] * y[iter-1][1][j+3] + A[6] * y[iter-1][2][j+1] + A[7] * y[iter-1][2][j+2] + A[8] * y[iter-1][2][j+3];
        float x4 = *(++xp) * a + *(++sp)  + A[0] * y[iter-1][0][j+2] + A[1] * y[iter-1][0][j+3] + A[2] * y[iter-1][0][j+4] + A[3] * y[iter-1][1][j+2] + A[4] * y[iter-1][1][j+3] + A[5] * y[iter-1][1][j+4] + A[6] * y[iter-1][2][j+2] + A[7] * y[iter-1][2][j+3] + A[8] * y[iter-1][2][j+4];
        if(fabs(x1)>1){
          if(x1<0){*(++bufp)=255;}
          else if(x1>0){*(++bufp)=0;}
        }
        else{*(++bufp)=(x1-1)*div;}
        if(fabs(x2)>1){
          if(x2<0){*(++bufp)=255;}
          else if(x2>0){*(++bufp)=0;}
        }
        else{*(++bufp)=(x2-1)*div;}
        if(fabs(x3)>1){
          if(x3<0){*(++bufp)=255;}
          else if(x3>0){*(++bufp)=0;}
        }
        else{*(++bufp)=(x3-1)*div;}
        if(fabs(x4)>1){
          if(x4<0){*(++bufp)=255;}
          else if(x4>0){*(++bufp)=0;}
        }
        else{*(++bufp)=(x4-1)*div;}
      }
      memcpy(&fb->buf[(i-iter)*w+1+w/2+1-patch-2],&buf8[1],((w-1)/2+patch-2)  * sizeof(uint8_t));
    }


  
  }

      xSemaphoreGive(semaphore2);
          vTaskDelete(NULL); 
}

// CNN tasks handler
void CNN(camera_fb_t *fb, params_t params){
      params.fb=fb;
      params.iterations=iterations;
      xTaskCreatePinnedToCore(CNN_right, "Task 2", 55000, &params ,3, NULL, 0);
      xTaskCreatePinnedToCore(CNN_left, "Task 1", 55000, &params , 5, NULL, 1);
      xSemaphoreTake(semaphore2, portMAX_DELAY);
      xSemaphoreTake(semaphore1, portMAX_DELAY);
}


static esp_err_t stream_handler(httpd_req_t *req)
{ 
  camera_fb_t *fb = NULL;

  esp_err_t res = ESP_OK;
  size_t _jpg_buf_len = 0;
  uint8_t *_jpg_buf = NULL; 

  char *part_buf[64];

  int frame_counter = 0;

  int64_t fr_start = 0;
  int64_t fr_ready = 0;
  int64_t fr_face = 0;
  int64_t fr_recognize = 0;
  int64_t fr_encode = 0;

  static int64_t last_frame = 0;

  
  if (!last_frame)
  {
    last_frame = esp_timer_get_time();
  }

  res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
  if (res != ESP_OK)
  {
    return res;
  }

  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");


  fb = esp_camera_fb_get();
  uint16_t w = fb->width;
  uint16_t h = fb->height;
  size_t len = fb->len;

  uint8_t iterations2 =iterations;
  float dt2=dt;
  params_t params,params2;

  fill_pc_array();


  int totalFrames=0;
  int64_t totalTime=0;
  int64_t mitja=0;
  


  float dtt = 1 - dt;


  esp_camera_fb_return(fb);
  fb = NULL;

  while (true)
  {
    fb = esp_camera_fb_get();
    if (!fb)
    {
      Serial.println("Camera capture failed");
      res = ESP_FAIL;
    }
    else
    { 
      frame_counter++;

      fr_start = esp_timer_get_time();
      fr_ready = fr_start;
      fr_face = fr_start;
      fr_encode = fr_start;
      fr_recognize = fr_start;
      

      if (iterations != iterations2){

        iterations2 = iterations;
        totalFrames=0;
        totalTime=0;
        mitja=0;

      }

     
      int64_t fr_st = esp_timer_get_time();


      // CNN processing function
      CNN(fb,params);

      int64_t fr_end = esp_timer_get_time();
      int64_t frame_time = fr_end - fr_st;
      frame_time /= 1000;


          
    totalTime+=frame_time;
    totalFrames++;
    mitja=totalTime/totalFrames;
    Serial.printf("%u ms\n", mitja);
    


      bool jpeg_converted = frame2jpg(fb, Q, &_jpg_buf, &_jpg_buf_len);
      esp_camera_fb_return(fb);
      fb = NULL;
      if (!jpeg_converted)
      {
        Serial.println("JPEG compression failed");
        res = ESP_FAIL;
      }
    }

    if (res == ESP_OK){res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));}

    if (res == ESP_OK) {
      size_t hlen = snprintf((char *)part_buf, 64, _STREAM_PART, _jpg_buf_len);
      res = httpd_resp_send_chunk(req, (const char *)part_buf, hlen);
    }

    if (res == ESP_OK){ res = httpd_resp_send_chunk(req, (const char *)_jpg_buf, _jpg_buf_len); }

  
    if (fb)
    {
      esp_camera_fb_return(fb);
      fb = NULL;
      _jpg_buf = NULL;
    }
    else if (_jpg_buf)
    {
      free(_jpg_buf);
      _jpg_buf = NULL;
    }
    
    if (res != ESP_OK)
    {
      break;
    }
    

  }

  last_frame = 0;
  return res;
}

static esp_err_t cmd_handler(httpd_req_t *req)
{
  char *buf;
  size_t buf_len;
  char variable[32] = {
      0,
  };
  char value[32] = {
      0,
  };

  buf_len = httpd_req_get_url_query_len(req) + 1;
  if (buf_len > 1)
  {
    buf = (char *)malloc(buf_len);
    if (!buf)
    {
      httpd_resp_send_500(req);
      return ESP_FAIL;
    }
    if (httpd_req_get_url_query_str(req, buf, buf_len) == ESP_OK)
    { //////////////////////////////////////////////////////////////////////////////////
      if (httpd_query_key_value(buf, "var", variable, sizeof(variable)) == ESP_OK && httpd_query_key_value(buf, "val", value, sizeof(value)) == ESP_OK)
      {
      }
      else
      {
        free(buf);
        httpd_resp_send_404(req);
        return ESP_FAIL;
      }
    }
    else
    {
      free(buf);
      httpd_resp_send_404(req);
      return ESP_FAIL;
    }
    free(buf);
  }
  else
  {
    httpd_resp_send_404(req);
    return ESP_FAIL;
  }

  // int val = atoi(value);//////////
  float val = atoff(value); //////////
  sensor_t *s = esp_camera_sensor_get();
  int res = 0;

  if (!strcmp(variable, "framesize"))
  {
    if (s->pixformat == PIXFORMAT_JPEG)
      res = s->set_framesize(s, (framesize_t)val);
  }
  else if (!strcmp(variable, "quality"))
    res = s->set_quality(s, val);
  else if (!strcmp(variable, "contrast"))
    res = s->set_contrast(s, val);
  else if (!strcmp(variable, "brightness"))
    res = s->set_brightness(s, val);
  else if (!strcmp(variable, "saturation"))
    res = s->set_saturation(s, val);
  else if (!strcmp(variable, "gainceiling"))
    res = s->set_gainceiling(s, (gainceiling_t)val);
  else if (!strcmp(variable, "colorbar"))
    res = s->set_colorbar(s, val);
  else if (!strcmp(variable, "awb"))
    res = s->set_whitebal(s, val);
  else if (!strcmp(variable, "agc"))
    res = s->set_gain_ctrl(s, val);
  else if (!strcmp(variable, "aec"))
    res = s->set_exposure_ctrl(s, val);
  else if (!strcmp(variable, "hmirror"))
    res = s->set_hmirror(s, val);
  else if (!strcmp(variable, "vflip"))
    res = s->set_vflip(s, val);
  else if (!strcmp(variable, "awb_gain"))
    res = s->set_awb_gain(s, val);
  else if (!strcmp(variable, "agc_gain"))
    res = s->set_agc_gain(s, val);
  else if (!strcmp(variable, "aec_value"))
    res = s->set_aec_value(s, val);
  else if (!strcmp(variable, "aec2"))
    res = s->set_aec2(s, val);
  else if (!strcmp(variable, "dcw"))
    res = s->set_dcw(s, val);
  else if (!strcmp(variable, "bpc"))
    res = s->set_bpc(s, val);
  else if (!strcmp(variable, "wpc"))
    res = s->set_wpc(s, val);
  else if (!strcmp(variable, "raw_gma"))
    res = s->set_raw_gma(s, val);
  else if (!strcmp(variable, "lenc"))
    res = s->set_lenc(s, val);
  else if (!strcmp(variable, "special_effect"))
    res = s->set_special_effect(s, val);
  else if (!strcmp(variable, "wb_mode"))
    res = s->set_wb_mode(s, val);
  else if (!strcmp(variable, "ae_level"))
    res = s->set_ae_level(s, val);

  else if (!strcmp(variable, "Q"))
  {
    Q = val;
  }
  else if (!strcmp(variable, "z"))
  {
    z = val;
  }
  else if (!strcmp(variable, "dt"))
  {
    dt = val;
  }
  else if (!strcmp(variable, "iterations"))
  {
    iterations = val;
    // justone=iterations==1;
  }
  else if (!strcmp(variable, "A0"))
  {
    A[0] = val;
    AA[0] = val;
  }
  else if (!strcmp(variable, "A1"))
  {
    A[1] = val;
    AA[1] = val;
  }
  else if (!strcmp(variable, "A2"))
  {
    A[2] = val;
    AA[2] = val;
  }
  else if (!strcmp(variable, "A3"))
  {
    A[3] = val;
    AA[3] = val;
  }
  else if (!strcmp(variable, "A4"))
  {
    A[4] = val;
    AA[4] = val;
  }
  else if (!strcmp(variable, "A5"))
  {
    A[5] = val;
    AA[5] = val;
  }
  else if (!strcmp(variable, "A6"))
  {
    A[6] = val;
    AA[6] = val;
  }
  else if (!strcmp(variable, "A7"))
  {
    A[7] = val;
    AA[7] = val;
  }
  else if (!strcmp(variable, "A8"))
  {
    A[8] = val;
    AA[8] = val;
  }
  else if (!strcmp(variable, "B0"))
  {
    B[0] = val;
  }
  else if (!strcmp(variable, "B1"))
  {
    B[1] = val;
  }
  else if (!strcmp(variable, "B2"))
  {
    B[2] = val;
  }
  else if (!strcmp(variable, "B3"))
  {
    B[3] = val;
  }
  else if (!strcmp(variable, "B4"))
  {
    B[4] = val;
  }
  else if (!strcmp(variable, "B5"))
  {
    B[5] = val;
  }
  else if (!strcmp(variable, "B6"))
  {
    B[6] = val;
  }
  else if (!strcmp(variable, "B7"))
  {
    B[7] = val;
  }
  else if (!strcmp(variable, "B8"))
  {
    B[8] = val;
  }
  else if (!strcmp(variable, "face_detect"))
  {
    detection_enabled = val;
    if (!detection_enabled)
    {
      recognition_enabled = 0;
    }
  }
  else if (!strcmp(variable, "face_enroll"))
    is_enrolling = val;
  else if (!strcmp(variable, "face_recognize"))
  {

    recognition_enabled = val;
    if (recognition_enabled)
    {
      detection_enabled = val;
    }
  }

  else
  {
    res = -1;
  }

  if (res)
  {
    return httpd_resp_send_500(req);
  }

  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  return httpd_resp_send(req, NULL, 0);
}

static esp_err_t status_handler(httpd_req_t *req)
{
  static char json_response[1024]; // response

  sensor_t *s = esp_camera_sensor_get();
  char *p = json_response; // json amb paràmetres
  *p++ = '{';

  p += sprintf(p, "\"framesize\":%u,", s->status.framesize);
  p += sprintf(p, "\"quality\":%u,", s->status.quality);
  p += sprintf(p, "\"brightness\":%d,", s->status.brightness);
  p += sprintf(p, "\"contrast\":%d,", s->status.contrast);
  p += sprintf(p, "\"saturation\":%d,", s->status.saturation);
  p += sprintf(p, "\"sharpness\":%d,", s->status.sharpness);
  p += sprintf(p, "\"special_effect\":%u,", s->status.special_effect);
  p += sprintf(p, "\"wb_mode\":%u,", s->status.wb_mode);
  p += sprintf(p, "\"awb\":%u,", s->status.awb);
  p += sprintf(p, "\"awb_gain\":%u,", s->status.awb_gain);
  p += sprintf(p, "\"aec\":%u,", s->status.aec);
  p += sprintf(p, "\"aec2\":%u,", s->status.aec2);
  p += sprintf(p, "\"ae_level\":%d,", s->status.ae_level);
  p += sprintf(p, "\"aec_value\":%u,", s->status.aec_value);
  p += sprintf(p, "\"agc\":%u,", s->status.agc);
  p += sprintf(p, "\"agc_gain\":%u,", s->status.agc_gain);
  p += sprintf(p, "\"gainceiling\":%u,", s->status.gainceiling);
  p += sprintf(p, "\"bpc\":%u,", s->status.bpc);
  p += sprintf(p, "\"wpc\":%u,", s->status.wpc);
  p += sprintf(p, "\"raw_gma\":%u,", s->status.raw_gma);
  p += sprintf(p, "\"lenc\":%u,", s->status.lenc);
  p += sprintf(p, "\"vflip\":%u,", s->status.vflip);
  p += sprintf(p, "\"hmirror\":%u,", s->status.hmirror);
  p += sprintf(p, "\"dcw\":%u,", s->status.dcw);
  p += sprintf(p, "\"colorbar\":%u,", s->status.colorbar);
  p += sprintf(p, "\"Q\":%u,", Q);                   //////////////////////////
  p += sprintf(p, "\"z\":%f,", z);                   //////////////////////////
  p += sprintf(p, "\"dt\":%f,", dt);                 //////////////////////////
  p += sprintf(p, "\"iterations\":%u,", iterations); //////////////////////////

  p += sprintf(p, "\"A0\":%f,", A[0]); //////////////////////////
  p += sprintf(p, "\"A1\":%f,", A[1]); //////////////////////////
  p += sprintf(p, "\"A2\":%f,", A[2]); //////////////////////////
  p += sprintf(p, "\"A3\":%f,", A[3]); //////////////////////////
  p += sprintf(p, "\"A4\":%f,", A[4]); //////////////////////////
  p += sprintf(p, "\"A5\":%f,", A[5]); //////////////////////////
  p += sprintf(p, "\"A6\":%f,", A[6]); //////////////////////////
  p += sprintf(p, "\"A7\":%f,", A[7]); //////////////////////////
  p += sprintf(p, "\"A8\":%f,", A[8]); //////////////////////////
  p += sprintf(p, "\"B0\":%f,", B[0]); //////////////////////////
  p += sprintf(p, "\"B1\":%f,", B[1]); //////////////////////////
  p += sprintf(p, "\"B2\":%f,", B[2]); //////////////////////////
  p += sprintf(p, "\"B3\":%f,", B[3]); //////////////////////////
  p += sprintf(p, "\"B4\":%f,", B[4]); //////////////////////////
  p += sprintf(p, "\"B5\":%f,", B[5]); //////////////////////////
  p += sprintf(p, "\"B6\":%f,", B[6]); //////////////////////////
  p += sprintf(p, "\"B7\":%f,", B[7]); //////////////////////////
  p += sprintf(p, "\"B8\":%f,", B[8]); //////////////////////////

  p += sprintf(p, "\"face_detect\":%u,", detection_enabled);
  p += sprintf(p, "\"face_enroll\":%u,", is_enrolling);
  p += sprintf(p, "\"face_recognize\":%u", recognition_enabled);

  *p++ = '}';
  *p++ = 0;
  httpd_resp_set_type(req, "application/json");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  return httpd_resp_send(req, json_response, strlen(json_response));
}

static esp_err_t index_handler(httpd_req_t *req)
{
  httpd_resp_set_type(req, "text/html");
  httpd_resp_set_hdr(req, "Content-Encoding", "gzip");
  sensor_t *s = esp_camera_sensor_get();
  if (s->id.PID == OV3660_PID)
  {
    return httpd_resp_send(req, (const char *)index_ov3660_html_gz, index_ov3660_html_gz_len);
  }
  return httpd_resp_send(req, (const char *)index_ov2640_html_gz, index_ov2640_html_gz_len);
}

void startCameraServer()
{

  httpd_config_t config = HTTPD_DEFAULT_CONFIG();

  httpd_uri_t index_uri = {
      .uri = "/",
      .method = HTTP_GET,
      .handler = index_handler,
      .user_ctx = NULL};

  httpd_uri_t status_uri = {
      .uri = "/status",
      .method = HTTP_GET,
      .handler = status_handler,
      .user_ctx = NULL};

  httpd_uri_t cmd_uri = {
      .uri = "/control",
      .method = HTTP_GET,
      .handler = cmd_handler,
      .user_ctx = NULL};

  httpd_uri_t stream_uri = {
      .uri = "/stream",
      .method = HTTP_GET,
      .handler = stream_handler,
      .user_ctx = NULL};

  

  Serial.printf("Starting web server on port: '%d'\n", config.server_port);
  if (httpd_start(&camera_httpd, &config) == ESP_OK)
  {
    httpd_register_uri_handler(camera_httpd, &index_uri);
    httpd_register_uri_handler(camera_httpd, &cmd_uri);
    httpd_register_uri_handler(camera_httpd, &status_uri);

  }

  config.server_port += 1;
  config.ctrl_port += 1;
  Serial.printf("Starting stream server on port: '%d'\n", config.server_port);
  if (httpd_start(&stream_httpd, &config) == ESP_OK)
  {
    httpd_register_uri_handler(stream_httpd, &stream_uri);
  }
}