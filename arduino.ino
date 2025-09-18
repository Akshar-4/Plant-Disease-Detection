#include "DHT.h"

int motor1pin1 = 2;
int motor1pin2 = 3;
int motor2pin1 = 4;
int motor2pin2 = 5;
int soilPin = A0;  

#define DHTPIN 6      
#define DHTTYPE DHT22
DHT dht(DHTPIN, DHTTYPE);

void setup() {
  pinMode(motor1pin1, OUTPUT);
  pinMode(motor1pin2, OUTPUT);
  pinMode(motor2pin1, OUTPUT);
  pinMode(motor2pin2, OUTPUT);

  Serial.begin(9600);
  dht.begin();

  Serial.println("Commands: f=forward, b=backward, l=left, r=right, s=stop");
}

void stopMotors() {
  digitalWrite(motor1pin1, LOW); digitalWrite(motor1pin2, LOW);
  digitalWrite(motor2pin1, LOW); digitalWrite(motor2pin2, LOW);
}

void forward() {
  digitalWrite(motor1pin1, HIGH); digitalWrite(motor1pin2, LOW);
  digitalWrite(motor2pin1, HIGH); digitalWrite(motor2pin2, LOW);
}

void backward() {
  digitalWrite(motor1pin1, LOW); digitalWrite(motor1pin2, HIGH);
  digitalWrite(motor2pin1, LOW); digitalWrite(motor2pin2, HIGH);
}

void turnLeft() {
  digitalWrite(motor1pin1, LOW); digitalWrite(motor1pin2, HIGH);
  digitalWrite(motor2pin1, HIGH); digitalWrite(motor2pin2, LOW);
}

void turnRight() {
  digitalWrite(motor1pin1, HIGH); digitalWrite(motor1pin2, LOW);
  digitalWrite(motor2pin1, LOW); digitalWrite(motor2pin2, HIGH);
}

void loop() {
  int soilValue = analogRead(soilPin); 
  float moisturePercent = map(soilValue, 1023, 0, 0, 100); 

  float humidity = dht.readHumidity();
  float temperature = dht.readTemperature();

  Serial.print(temperature);
  Serial.print(",");
  Serial.print(humidity);
  Serial.print(",");
  Serial.println(moisturePercent);

  delay(2000); 

  if (Serial.available() > 0) {
    char command = Serial.read();

    if (command == 'f') forward();
    else if (command == 'b') backward();
    else if (command == 'l') turnLeft();
    else if (command == 'r') turnRight();
    else if (command == 's') stopMotors();
  }
}
