#include <SoftwareSerial.h>

// Pinos virtuais da Serial (RX no pino 2, TX no pino 3)
SoftwareSerial bleSerial(2, 3); 

void setup() {
  // Serial de depuração no USB/Monitor Serial
  Serial.begin(9600);
  
  // Comunicação com o módulo Bluetooth (Taxa padrão 9600 bps)
  bleSerial.begin(38400);
  
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);

  Serial.println("Arduino pronto! Aguardando mensagens da MainActivity...");
}

void loop() {
  // Verifica se chegou mensagem do App Android
  
  if (bleSerial.available()) {
    String mensagem = bleSerial.readString();
    mensagem.trim();
    
    if (mensagem.length() > 0) {
      Serial.print("[App -> Arduino]: ");
      Serial.println(mensagem);
    }
  }
}