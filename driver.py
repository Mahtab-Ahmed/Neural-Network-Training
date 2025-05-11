import msgParser
import carState
import carControl
import pandas as pd
import numpy as np
import ast
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import os
import pickle
import joblib

class Driver(object):
    '''
    A driver object for the SCRC that uses a Random Forest model for predictions
    '''

    def __init__(self, stage):
        '''Constructor'''
        self.WARM_UP = 0
        self.QUALIFYING = 1
        self.RACE = 2
        self.UNKNOWN = 3
        self.stage = stage
        
        self.parser = msgParser.MsgParser()
        self.state = carState.CarState()
        self.control = carControl.CarControl()
        
        self.steer_lock = 0.785398
        self.max_speed = 100
        self.prev_rpm = None
        
        # Feature list - what we'll use for prediction
        self.features = [
            'angle', 'trackPos', 'speedX', 'speedY', 'speedZ', 'rpm', 'gear',
            'distRaced', 'fuel'
        ]
        
        # Models for steering, gear, and acceleration
        self.steer_model = None
        self.gear_model = None
        self.accel_model = None
        
        # Try to load existing models or train new ones
        self.load_or_train_models()
        
    def load_or_train_models(self):
        """Only load pre-trained models from disk"""
        try:
            print("[INFO] Loading existing models...")
            self.steer_model = joblib.load("models/steer_model.pkl")
            self.gear_model = joblib.load("models/gear_model.pkl")
            self.accel_model = joblib.load("models/accel_model.pkl")
            print("[✓] Models loaded successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to load models: {e}")
            print("Using fallback driving logic.")


    

    def train_models_from_preprocessed(self, preprocessed_csv_path):
        """Train models from already preprocessed CSV and save them"""
        try:
            data = pd.read_csv(preprocessed_csv_path)

            target_columns = ['steer', 'accel', 'brake']
            feature_columns = [col for col in data.columns if col not in target_columns]

            print(f"[INFO] Training on features: {feature_columns}")
            print(f"[INFO] Targets: {target_columns}")

            self.models = {}
            for target in target_columns:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(data[feature_columns], data[target])
                self.models[target] = model
                pickle.dump(model, open(f'{target}_model.pkl', 'wb'))
                print(f"[✓] Trained and saved model for '{target}'")
            
           
            
        except Exception as e:
            print(f"[ERROR] Failed to train models from preprocessed data: {e}")


    
    def init(self):
        '''Return init string with rangefinder angles'''
        self.angles = [0 for x in range(19)]
        
        for i in range(5):
            self.angles[i] = -90 + i * 15
            self.angles[18 - i] = 90 - i * 15
        
        for i in range(5, 9):
            self.angles[i] = -20 + (i-5) * 5
            self.angles[18 - i] = 20 - (i-5) * 5
        
        return self.parser.stringify({'init': self.angles})
    
    def drive(self, msg):
        self.state.setFromMsg(msg)
        
        if self.steer_model and self.gear_model and self.accel_model:
            self.model_drive()
        else:
            # Fallback to original driving logic
            self.steer()
            self.gear()
            self.speed()
        
        return self.control.toMsg()
    
    def set_throttle(self, value):
        if value <= 0:
            self.control.setBrake(-value)
            self.control.setAccel(0)
        else:
            self.control.setAccel(value)
            self.control.setBrake(0)

    def model_drive(self):
        """Use trained models to drive the car"""
        # Extract features from the current state
        if self.state.getDistRaced() < 5:
            self.control.setAccel(1.0)
            self.control.setBrake(0.0)
            self.control.setSteer(0.0)
            self.control.setGear(1)
            return self.control.toMsg()

        features = self.extract_features()
        
        # Make predictions if we have all needed features
        try:
            # Predict steering
            steer_value = self.steer_model.predict([features])[0]
            self.control.setSteer(np.clip(steer_value, -1.0, 1.0))
            
            # Predict gear
            gear_value = int(round(self.gear_model.predict([features])[0]))
            # Ensure gear is valid (between -1 and 6)
            gear_value = max(-1, min(6, gear_value))
            self.control.setGear(gear_value)
            
            # Predict acceleration
            accel_value = self.accel_model.predict([features])[0]
            self.control.set_throttle(np.clip(accel_value, 0.0, 1.0))
            
          
            
            # Store the current RPM for the next cycle
            self.prev_rpm = self.state.getRpm()
            
            # Print occasional status
            if self.state.getDistRaced() % 100 < 1:
                print(f"Distance: {self.state.getDistRaced():.1f}, Speed: {self.state.getSpeedX():.1f}, " + 
                      f"Steering: {steer_value:.3f}, Accel: {accel_value:.2f}, Gear: {gear_value}")
            
        except Exception as e:
            print(f"Error during model prediction: {e}")
            # Fallback to original driving logic
            self.steer()
            self.gear()
            self.speed()
    
    def extract_features(self):
        """Extract features from the current state to feed into the model"""
        s = self.state  # shorthand

        feature_dict = {
            'angle': s.angle,
            'distFromStart': s.getDistFromStart(),
            'distRaced': s.getDistRaced(),
            'fuel': s.getFuel(),
            'gear': s.getGear(),
            'racePos': s.getRacePos(),
            'rpm': s.getRpm(),
            'speedX': s.getSpeedX(),
            'speedY': s.getSpeedY(),
            'speedZ': s.getSpeedZ(),
            'trackPos': s.trackPos,
            'z': s.z,  # assuming this is a valid attribute
            'accel': s.getAccel(),
            'brake': s.getBrake(),
            'gear.1': s.getGear(),  # duplicate, but kept if used in training
            'steer': s.getSteer(),
            'clutch': s.getClutch(),
            'focus.1': s.getFocus(),  # assuming same logic
            'meta': s.getMeta(),      # optional, if used
        }

        # Expand track sensors
        track = s.getTrack()
        for i in range(len(track)):
            feature_dict[f'track_{i}'] = track[i]

        # Expand opponents
        opponents = s.getOpponents()
        for i in range(len(opponents)):
            feature_dict[f'opponents_{i}'] = opponents[i]

        # Expand wheel spin velocity
        wheel = s.getWheelSpinVel()
        for i in range(len(wheel)):
            feature_dict[f'wheelSpinVel_{i}'] = wheel[i]

        # Expand focus sensors
        focus = s.getFocus()
        for i in range(len(focus)):
            feature_dict[f'focus_{i}'] = focus[i]

        # Convert to list in the order of self.features
        feature_values = []
        for feature in self.features:
            feature_values.append(feature_dict.get(feature, 0))  # default to 0 if missing

        return feature_values

    
    # Original methods kept as fallback
    def steer(self):
        angle = self.state.angle
        dist = self.state.trackPos
        
        self.control.setSteer((angle - dist*0.5)/self.steer_lock)
    
    def gear(self):
        rpm = self.state.getRpm()
        gear = self.state.getGear()
        
        if self.prev_rpm == None:
            up = True
        else:
            if (self.prev_rpm - rpm) < 0:
                up = True
            else:
                up = False
        
        if up and rpm > 7000:
            gear += 1
        
        if not up and rpm < 3000:
            gear -= 1
        
        self.control.setGear(gear)
        self.prev_rpm = rpm
    
    def speed(self):
        speed = self.state.getSpeedX()
        accel = self.control.getAccel()
        
        if speed < self.max_speed:
            accel += 0.1
            if accel > 1:
                accel = 1.0
        else:
            accel -= 0.1
            if accel < 0:
                accel = 0.0
        
        self.control.setAccel(accel)
            
    def onShutDown(self):
        pass
    
    def onRestart(self):
        pass