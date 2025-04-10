#define LED1 26  
#define LED2 27  
#define LED3 32  
#define BUZZER 33  

String inputString = "";
bool stringComplete = false;

void setup() {
  Serial.begin(115200);
  pinMode(LED1, OUTPUT);
  pinMode(LED2, OUTPUT);
  pinMode(LED3, OUTPUT);
  pinMode(BUZZER, OUTPUT);

  digitalWrite(LED1, LOW);
  digitalWrite(LED2, LOW);
  digitalWrite(LED3, LOW);
  digitalWrite(BUZZER, LOW);

  inputString.reserve(200);
}

void loop() {
  if (stringComplete) {
    inputString.trim();
    Serial.print("Received emotion: ");
    Serial.println(inputString);

    handleEmotion(inputString);
    inputString = "";
    stringComplete = false;
  }
}

void handleEmotion(String emotion) {
    // Turn off everything first
    digitalWrite(LED1, LOW);
    digitalWrite(LED2, LOW);
    digitalWrite(LED3, LOW);
    digitalWrite(BUZZER, LOW);
  
    if (emotion == "surprise") {
      blinkAll(2, 300);
      buzz(1, 500);
      delay(1000);
    } else if (emotion == "sadness") {
      digitalWrite(LED1, HIGH);
      delay(1000);
    } else if (emotion == "love") {
      digitalWrite(LED1, HIGH);
      digitalWrite(LED2, HIGH);
      delay(1000);
    } else if (emotion == "anger") {
      digitalWrite(LED1, HIGH);
      digitalWrite(LED2, HIGH);
      digitalWrite(LED3, HIGH);
      digitalWrite(BUZZER, HIGH);
      delay(1000);
    } else if (emotion == "fear") {
      digitalWrite(LED3, HIGH);
      buzz(2, 300);
    } else if (emotion == "joy") {
      digitalWrite(LED1, HIGH);
      digitalWrite(LED2, HIGH);
      digitalWrite(LED3, HIGH);
      delay(1000);
    }
    
    // Turn everything off after action
  digitalWrite(LED1, LOW);
  digitalWrite(LED2, LOW);
  digitalWrite(LED3, LOW);
  digitalWrite(BUZZER, LOW);
}

void blinkAll(int times, int delayTime) {
  for (int i = 0; i < times; i++) {
    digitalWrite(LED1, HIGH);
    digitalWrite(LED2, HIGH);
    digitalWrite(LED3, HIGH);
    delay(delayTime);
    digitalWrite(LED1, LOW);
    digitalWrite(LED2, LOW);
    digitalWrite(LED3, LOW);
    delay(delayTime);
  }
}

void buzz(int times, int duration) {
  for (int i = 0; i < times; i++) {
    digitalWrite(BUZZER, HIGH);
    delay(duration);
    digitalWrite(BUZZER, LOW);
    delay(duration);
  }
}
void serialEvent() {
    while (Serial.available()) {
      char inChar = (char)Serial.read();
      if (inChar == '\n') {
        stringComplete = true;
      } else {
        inputString += inChar;
  }
  }
}