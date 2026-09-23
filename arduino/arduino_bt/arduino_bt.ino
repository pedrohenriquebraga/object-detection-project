#include <SoftwareSerial.h>

SoftwareSerial bleSerial(2, 3); 

void setup() {
  Serial.begin(9600);
  
  bleSerial.begin(38400);
  
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);

  Serial.println("Arduino pronto! Aguardando mensagens da MainActivity...");
}

void loop() {
  
  if (bleSerial.available()) {
    String mensagem = bleSerial.readString();
    mensagem.trim();
    
    if (mensagem.length() > 0) {
      Serial.print("[App -> Arduino]: ");
      Serial.println(mensagem);
    }
  }
}