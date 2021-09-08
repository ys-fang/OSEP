
#include<WiFi.h>

//#include <Servo.h>
#include "DHTStable.h"
#include <Wire.h> 
#include <LiquidCrystal_I2C.h>
//oled
#include <string.h>
#include <Arduino.h>
#include <U8g2lib.h>

//ws2812
#include <Adafruit_NeoPixel.h>
#ifdef __AVR__
 #include <avr/power.h> // Required for 16 MHz Adafruit Trinket
#endif

#define NUMPIXELS 12 // Popular NeoPixel ring size
//Adafruit_NeoPixel pixels(NUMPIXELS, 5, NEO_GRB + NEO_KHZ800);

U8G2_SSD1306_128X64_NONAME_1_HW_I2C u8g2(U8G2_R0, /* reset=*/ U8X8_PIN_NONE);
//oled end

//pwm
// setting PWM properties
const int freq = 5000;
const int ledChannel = 0;
const int resolution = 8;
//

LiquidCrystal_I2C lcd(0x27, 16, 2);  //設定LCD
DHTStable DHT;
//Servo myservo;  // create servo object to control a servo


//const char* ssid = ""; // Your WiFi SSID
//const char* password = ""; // Your WiFi Password
//wifi
///const uint16_t port = 3000;
//ESP8266WiFiMulti WiFiMulti;



/*
const int outPin0 =  0;
const int outPin1 =  1;
const int outPin2 =  2;
const int outPin3 =  3;
const int outPin5 =  5;       // the number of the LED pin
const int outPin4 =  4;
const int outPin12 =  12;
const int outPin13 =  13;
const int outPin14 =  14;
const int outPin15 =  15;
*/
// variable for storing the pushbutton status

char* serialString()
{
  //static char str[21]; // For strings of max length=20
  static char str[21]; // For strings of max length=20
  if (!Serial.available()) return NULL;
  delay(32); // wait for all characters to arrive
  memset(str,0,sizeof(str)); // clear str
  byte count=0;
  while (Serial.available())
  {
    char c=Serial.read();
    //if (c>=32 && count<sizeof(str)-1)
    //c最大35
    //{
      str[count]=c;
      count++;
    //}
  }
  str[count]='\0'; // make it a zero terminated string
  return str;
}

void setup() {
  Serial.begin(115200);
  lcd.init(); //初始化LCD 
  lcd.begin(16, 2); //初始化 LCD，代表我們使用的LCD一行有16個字元，共2行。
  //lcd.backlight(); //開啟背光
  //oled
  u8g2.begin();
  u8g2.enableUTF8Print();  //啟用UTF8文字的功能  
  ledcSetup(ledChannel, freq, resolution);
  //digital 2 esp32 led燈
  //digital out 1-5 12-33 37
  //gpio 15-19 25 26 ok
  //wifi
  WiFi.mode(WIFI_STA); //設置WiFi模式
  
  //ws2812
  #if defined(__AVR_ATtiny85__) && (F_CPU == 16000000)
  clock_prescale_set(clock_div_1);
  #endif
  //pixels.begin(); // INITIALIZE NeoPixel strip object (REQUIRED)
  
}

void loop() {
    static boolean needPrompt=true;
    char* inputData;
    if (needPrompt)
    {
    //Serial.print("Please enter inputs and press enter at the end:\n");
      needPrompt=false;
    }
    inputData= serialString();
    //inputData = Serial.read();
    
    if (inputData!=NULL)
     {
      //Serial.println(inputData);
      char* commandString = strtok(inputData, "#"); 
      //Serial.println(commandString);
      char* inputPin = strtok(NULL, "#");
      //Serial.println(inputPin);
      //取出第3個值
      char* inputValue = strtok(NULL, "#");
      //Serial.println(inputValue);
      //取出第4個值
      char* inputTime =strtok(NULL, "#");
      //Serial.println(inputTime);

      //ws2812
      if(strcmp(commandString, "ws") == 0){
        int r = atoi(strtok(inputValue,","));
        int g = atoi(strtok(NULL, ","));
        int b = atoi(strtok(NULL, ","));
        Adafruit_NeoPixel pixels(NUMPIXELS, atoi(inputPin), NEO_GRB + NEO_KHZ800);
        pixels.begin();
        //pixels.clear();
        pixels.setPixelColor(atoi(inputTime), pixels.Color(r, g, b));
        
        //pixels.setPixelColor(5, pixels.Color(r, g, b));
        pixels.show(); 
        
      }
      
      //wifi
      if(strcmp(commandString, "w") == 0){
         //WiFi.begin(inputPin,inputValue);
         //char * password = "0936469648";
         //Serial.println(inputPin);
         //Serial.println(inputValue);
         WiFi.begin(inputPin,inputValue);
         byte count=0;
         while (WiFi.status() != WL_CONNECTED)
         {
            delay(500);
            //Serial.print(".");
            count ++;
            if (count >15){
              break;
            }
         }
         //Serial.println(WiFi.RSSI()); //讀取WiFi強度
         Serial.println(WiFi.localIP());
         
         
      }
      
      if(strcmp(commandString, "HC-SR04")== 0){
       long duration, cm; 
       int trigPin = atoi(inputPin);
       int echoPin = atoi(inputValue);
       pinMode(trigPin, OUTPUT);        // 定義輸入及輸出 
       pinMode(echoPin, INPUT);
       digitalWrite(trigPin, LOW);
       delayMicroseconds(5);
       //digitalWrite(trigPin, HIGH);     // 給 Trig 高電位，持續 10微秒  
       digitalWrite(trigPin, HIGH);     // 給 Trig 高電位，持續 10微秒  
       delayMicroseconds(10);
       digitalWrite(trigPin, LOW);
       //pinMode(echoPin, INPUT);             // 讀取 echo 的電位
       pinMode(echoPin, INPUT);             // 讀取 echo 的電位
       duration = pulseIn(echoPin, HIGH);   // 收到高電位時的時間
       cm = (duration/2) / 29.1;         // 將時間換算成距離 cm
       Serial.println(cm);        
    }
     //oled 16x2
    //format: l#string#row
    if(strcmp(commandString, "o") == 0) {
        u8g2.setFont(u8g2_font_unifont_t_chinese1); //使用字型
        u8g2.firstPage();
        do {
          
          if(atoi(inputValue) == 0){
            u8g2.setCursor(0, 14);
          }
          if(atoi(inputValue) == 1 ){
            u8g2.setCursor(0, 35);
          }
          if(atoi(inputValue) == 2 ){
            u8g2.setCursor(0, 60);
          }
          //u8g2.setCursor(0, 35);
          u8g2.print(inputPin);
        }while ( u8g2.nextPage() );
            //delay(1000);
    }
    
    //lcd 16x2
    //format: l#string#row
    if(strcmp(commandString, "l_clear") == 0) {
      lcd.backlight(); //開啟背光
      lcd.clear();
      lcd.noBacklight(); // 關閉背光
    }
    if(strcmp(commandString, "l") == 0) {
      lcd.backlight(); //開啟背光
          //Serial.println(inputPin);
          if(atoi(inputValue) == 0){
            lcd.setCursor(0,0);
          }else{
            lcd.setCursor(0,1);
          }
          lcd.print(inputPin);  
    }
      
      //dht11
      
      if(strcmp(commandString, "dht11Read") == 0){
      int chk = DHT.read11(atoi(inputPin));
      
      if( atoi(inputValue) == 1 ){
          Serial.println(DHT.getTemperature(), 1);
        }else{
          Serial.println(DHT.getHumidity(), 1);
        }
      
      }
      
      //tone
      /*if(strcmp(commandString, "tonePlay") == 0){
        int toneTime = atoi(inputTime);
        int tonePin = atoi(inputPin);
        tone(tonePin, atoi(inputValue),toneTime);
        //tone(4,110,1000);
        delay(toneTime);
        noTone(tonePin);
        delay(10);
      }*/
      //類比讀取
      
      if(strcmp(commandString, "analogRead") == 0){
        int int_inputPin = atoi(inputPin);
        pinMode(int_inputPin, INPUT);
        Serial.println(analogRead(int_inputPin));
      }
      //數位讀取
      if(strcmp(commandString, "digitalRead") == 0){
        int int_inputPin = atoi(inputPin);
        pinMode(int_inputPin, INPUT);
        Serial.println(digitalRead(int_inputPin));
      }
      //pwm類比寫入
      if(strcmp(commandString, "pwm") == 0){
           ledcAttachPin(atoi(inputPin), ledChannel);
           ledcWrite(ledChannel, atoi(inputValue));
           //analogWrite(atoi(inputPin),atoi(inputValue));
      }
     //Touch T0 T3 T4 T7 T9
     if(strcmp(commandString, "touchRead") == 0){
        if(strcmp(inputPin, "T0") == 0){
          Serial.println(touchRead(T0));
        }
        if(strcmp(inputPin, "T1") == 0){
          Serial.println(touchRead(T1));
        }
        if(strcmp(inputPin, "T2") == 0){
          Serial.println(touchRead(T2));
        }
        if(strcmp(inputPin, "T3") == 0){
          Serial.println(touchRead(T3));
        }
        if(strcmp(inputPin, "T4") == 0){
          Serial.println(touchRead(T4));
        }
        if(strcmp(inputPin, "T5") == 0){
          Serial.println(touchRead(T5));
        }
        if(strcmp(inputPin, "T6") == 0){
          Serial.println(touchRead(T6));
        }
        if(strcmp(inputPin, "T7") == 0){
          Serial.println(touchRead(T7));
        }
        if(strcmp(inputPin, "T8") == 0){
          Serial.println(touchRead(T8));
        }
        if(strcmp(inputPin, "T9") == 0){
          Serial.println(touchRead(T9));
        }
      }
    //數位寫入
      if(strcmp(commandString, "digitalWrite") == 0){
        int digitalPin = atoi(inputPin);
        pinMode(digitalPin, OUTPUT);
        digitalWrite(digitalPin,atoi(inputValue));
      }
      //Serial.println(inputData);
      needPrompt=true;
      
     }
    //delay(10);
    
  }
