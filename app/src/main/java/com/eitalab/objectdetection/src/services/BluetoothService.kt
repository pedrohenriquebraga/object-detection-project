package com.eitalab.objectdetection.src.services

import android.Manifest
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothManager
import android.content.Context
import android.content.Intent
import android.widget.Toast
import androidx.annotation.RequiresPermission
import androidx.core.app.ActivityCompat.startActivityForResult

class BluetoothService(private val context: Context) {

    private lateinit var bluetoothManager: BluetoothManager;
    private var bluetoothAdapter: BluetoothAdapter? = null;

    init {
        runBluetoothService()
    }

    private fun runBluetoothService() {
        bluetoothManager = context.getSystemService(Context.BLUETOOTH_SERVICE) as BluetoothManager
        bluetoothAdapter = bluetoothManager.adapter

        if (bluetoothAdapter == null) {
            Toast.makeText(context, "O bluetooth não está disponível", Toast.LENGTH_SHORT).show()
        }
    }

    @RequiresPermission(Manifest.permission.BLUETOOTH_CONNECT)
    fun turnOnBluetooth() {
        if (bluetoothAdapter?.isEnabled == false) {
            val enableBtIntent = Intent(BluetoothAdapter.ACTION_REQUEST_ENABLE)
            context.startActivity(enableBtIntent)
        }
    }
}