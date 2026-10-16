package com.doorcounter.app;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;

/** Invisible launcher entry: starts the counter service and closes right away. */
public final class MainActivity extends Activity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        startService(new Intent(this, CounterService.class));
        finish();
    }
}
