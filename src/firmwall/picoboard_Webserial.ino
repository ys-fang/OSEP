char* serialString()
{
  static char str[21]; // For strings of max length=20
  if (!Serial.available()) return NULL;
  delay(10); // wait for all characters to arrive
  memset(str,0,sizeof(str)); // clear str
  byte count=0;
  while (Serial.available())
  {
    char c=Serial.read();
    if (c>=32 && count<sizeof(str)-1)
    {
      str[count]=c;
      count++;
    }
  }
  str[count]='\0'; // make it a zero terminated string
  return str;
}


void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);

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
  
  if (inputData!=NULL)
  {
  //[1J4-A1,2J3-A2,3J2-A4,4按鈕D2,J1-A5，6光敏A3,聲音micA3,8滑桿(A0)]
    Serial.print(analogRead(1));
    Serial.print(",");
    Serial.print(analogRead(2));
    Serial.print(",");
    Serial.print(analogRead(4));
    Serial.print(",");
    Serial.print(digitalRead(21));
    //Serial.print(analogRead(7));
    Serial.print(",");
    Serial.print(analogRead(5));
    Serial.print(",");
    Serial.print(analogRead(6));
    Serial.print(",");
    Serial.print(analogRead(3));
    Serial.print(",");
    Serial.println(analogRead(0));
    inputData = NULL;
  }
  needPrompt=true;
    
}
