package com.eitalab.objectdetection.src.services

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
import android.bluetooth.*
import android.bluetooth.le.BluetoothLeScanner
import android.bluetooth.le.ScanCallback
import android.bluetooth.le.ScanResult
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.widget.Toast
import androidx.annotation.RequiresApi
import androidx.annotation.RequiresPermission
import androidx.core.app.ActivityCompat
import java.util.UUID

class BluetoothService(private val context: Context) {
    interface OnDeviceFoundListener {
        fun onDeviceFound(device: BluetoothDevice)
        fun onScanFinished()
    }

    private var isScanning = false
    private val handler = Handler(Looper.getMainLooper())
    private val SCAN_PERIOD: Long = 5000
    private val foundDeviceAddresses = mutableSetOf<String>()
    private var bluetoothManager: BluetoothManager? = null
    private var bluetoothAdapter: BluetoothAdapter? = null
    private var bluetoothGatt: BluetoothGatt? = null
    private var bluetoothLeScanner: BluetoothLeScanner? = null


    // IMPORTANTE: Substitua estes UUIDs pelos do seu dispositivo BLE (Ex: HM-10, ESP32, Arduino)
    // Estes são exemplos comuns para módulos UART BLE
    private val SERVICE_UUID: UUID = UUID.fromString("0000ffe0-0000-1000-8000-00805f9b34fb")
    private val CHAR_UUID: UUID = UUID.fromString("0000ffe1-0000-1000-8000-00805f9b34fb")

    var isConnected = false

    private val gattCallback = object : BluetoothGattCallback() {
        @SuppressLint("MissingPermission")
        override fun onConnectionStateChange(gatt: BluetoothGatt?, status: Int, newState: Int) {
            if (newState == BluetoothProfile.STATE_CONNECTED) {
                Log.d("BLE", "Conectado ao servidor GATT.")
                isConnected = true
                gatt?.discoverServices()

            } else if (newState == BluetoothProfile.STATE_DISCONNECTED) {
                Log.d("BLE", "Desconectado do servidor GATT.")
                isConnected = false
                bluetoothGatt = null
            }
        }

        override fun onServicesDiscovered(gatt: BluetoothGatt?, status: Int) {
            if (status == BluetoothGatt.GATT_SUCCESS) {
                Log.d("BLE", "Serviços descobertos.")
                sendMessage("Mensagem enviada via app Android 😊\n")
            }
        }
    }

    init {
        setupBluetooth()
    }

    private fun setupBluetooth() {
        bluetoothManager = context.getSystemService(Context.BLUETOOTH_SERVICE) as BluetoothManager
        bluetoothAdapter = bluetoothManager?.adapter
        bluetoothLeScanner = bluetoothAdapter?.bluetoothLeScanner

        if (bluetoothAdapter == null) {
            Toast.makeText(context, "O bluetooth não está disponível", Toast.LENGTH_SHORT).show()
        }
    }

    @RequiresPermission(Manifest.permission.BLUETOOTH_SCAN)
    fun scanLeDevice(listener: OnDeviceFoundListener) {
        val scanCallback = object : ScanCallback() {
            override fun onScanResult(callbackType: Int, result: ScanResult) {
                val device = result.device
                val deviceAddress = device.address
                if (!foundDeviceAddresses.contains(deviceAddress)) {
                    foundDeviceAddresses.add(deviceAddress)
                    listener.onDeviceFound(device)
                }
            }

            override fun onScanFailed(errorCode: Int) {
                super.onScanFailed(errorCode)
                isScanning = false
                listener.onScanFinished()
            }
        }

        if (!isScanning) {
            handler.postDelayed({
                isScanning = false
                bluetoothLeScanner?.stopScan(scanCallback)
            }, SCAN_PERIOD)
            isScanning = true
            bluetoothLeScanner?.startScan(scanCallback)
        } else {
            isScanning = false
            bluetoothLeScanner?.stopScan(scanCallback)
        }
    }

    @RequiresPermission(Manifest.permission.BLUETOOTH_CONNECT)
    private fun getPairedDeviceList(): Set<BluetoothDevice>? {
        return bluetoothAdapter?.bondedDevices
    }

    @RequiresPermission(allOf = [Manifest.permission.BLUETOOTH_SCAN, Manifest.permission.BLUETOOTH_CONNECT])
    fun connectAssistiveDevice(newMacAddress: String = "") {
        if (isConnected || bluetoothGatt != null) return

        val storage = StorageService(context as Activity)
        val macAddress = storage.getData("assistive_device_mac")

        if (macAddress.isNotEmpty() && newMacAddress.isEmpty()) {
            val pairedDevices = getPairedDeviceList()
            val targetDevice = pairedDevices?.find { it.address == macAddress }

            val device = targetDevice ?: bluetoothAdapter?.getRemoteDevice(macAddress)

            device?.let {
                bluetoothGatt = it.connectGatt(context, true, gattCallback)
            }
        } else {
            val device = bluetoothAdapter?.getRemoteDevice(newMacAddress)

            device?.let {
                bluetoothGatt = it.connectGatt(context, true, gattCallback)
                storage.saveData("assistive_device_mac", newMacAddress)
            }
        }
    }

    @SuppressLint("MissingPermission")
    fun sendMessage(message: String) {
        if (!isConnected || bluetoothGatt == null) return

        val service = bluetoothGatt?.getService(SERVICE_UUID)
        val characteristic = service?.getCharacteristic(CHAR_UUID)

        if (characteristic != null) {
            characteristic.setValue(message.toByteArray())

            characteristic.writeType = BluetoothGattCharacteristic.WRITE_TYPE_DEFAULT

            val success = bluetoothGatt?.writeCharacteristic(characteristic) ?: false

            if (!success) {
                Log.e("BLE", "Falha ao iniciar a escrita")
            }
        } else {
            Log.e("BLE", "Serviço ou Característica não encontrados. Verifique os UUIDs.")
        }
    }

    fun turnOnBluetooth() {
        if (bluetoothAdapter?.isEnabled == false) {
            val enableBtIntent = Intent(BluetoothAdapter.ACTION_REQUEST_ENABLE)
            if (ActivityCompat.checkSelfPermission(
                    context,
                    Manifest.permission.BLUETOOTH_CONNECT
                ) != PackageManager.PERMISSION_GRANTED
            ) {
                Toast.makeText(context, "Permissão de Conexão ao Bluetooth não concedida", Toast.LENGTH_SHORT).show()
                return
            }
            context.startActivity(enableBtIntent)
        }
    }

    fun onConnect() {
        this.sendMessage("Olá Mundo!")
    }

    @SuppressLint("MissingPermission")
    fun close() {
        bluetoothGatt?.close()
        bluetoothGatt = null
        isConnected = false
    }
}