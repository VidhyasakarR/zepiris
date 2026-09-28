package com.loadshare.lsncapture;

import androidx.activity.ComponentActivity;
import androidx.annotation.NonNull;
import androidx.lifecycle.ViewModelProvider;
import androidx.lifecycle.ViewModelStore;
import androidx.lifecycle.viewmodel.CreationExtras;

/**
 * Java shim between CaptureActivity and ComponentActivity.
 *
 * Host apps may force androidx.lifecycle:lifecycle-viewmodel below 2.6 (titan-rider-app forces
 * 2.5.1) while androidx.activity 1.9.x is compiled against 2.6+. In 2.6 the ViewModel owner
 * interfaces became Kotlin properties, so the Kotlin compiler no longer sees ComponentActivity's
 * members as implementing the old Java getters and rejects any Kotlin subclass
 * ("does not implement abstract member getViewModelStore"). Re-declaring the getters in Java
 * satisfies both interface shapes; behaviour is unchanged (every call delegates to super).
 */
public abstract class LsnBaseActivity extends ComponentActivity {

    @NonNull
    @Override
    public ViewModelStore getViewModelStore() {
        return super.getViewModelStore();
    }

    @NonNull
    @Override
    public ViewModelProvider.Factory getDefaultViewModelProviderFactory() {
        return super.getDefaultViewModelProviderFactory();
    }

    @NonNull
    @Override
    public CreationExtras getDefaultViewModelCreationExtras() {
        return super.getDefaultViewModelCreationExtras();
    }
}
