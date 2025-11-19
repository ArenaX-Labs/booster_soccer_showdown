# Copyright (c) 2024, The SAI Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard controller for SE(3) control."""

import numpy as np
from collections.abc import Callable
import glfw
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
import threading
import mujoco

try:
    from pynput import keyboard as pynput_keyboard
except Exception:
    pynput_keyboard = None


class Se3Keyboard:
    """A keyboard controller for sending SE(3) commands as delta velocity.

    This class is designed to provide a keyboard controller for humanoid robot.
    It uses GLFW keyboard interface to listen to keyboard events and map them to robot's
    task-space commands.

    The command comprises of:

    * delta vel: a 3D vector of (x, y) in meter/sec and z in rad/sec.

    Key bindings:
        ============================== ================= =================
        Description                    Key (+ve axis)    Key (-ve axis)
        ============================== ================= =================
        Move along x-axis              W                 S
        Move along y-axis              A                 D
        Rotate along z-axis            Q                 E
        Reset commands                 L
        Reset environment              P
        ============================== ================= =================
    """

    def __init__(self, renderer: MujocoRenderer, pos_sensitivity, rot_sensitivity):
        """Initialize the keyboard layer.

        Args:
            env: The Mujoco environment
            pos_sensitivity: Magnitude of input position command scaling.
            rot_sensitivity: Magnitude of scale input rotation commands scaling.
        """
        self._delta_vel = np.zeros(3)  # (x, y, yaw)

        self.rot_sensitivity = rot_sensitivity
        self.pos_sensitivity = pos_sensitivity
        self._should_quit = False

        # dictionary for additional callbacks
        self._additional_callbacks = dict()

        # store the viewer
        self._viewer = renderer._get_viewer("human")

        if hasattr(self._viewer, "_key_callback"):
            self._original_key_callback = self._viewer._key_callback
        else:
            self._original_key_callback = None

        # register keyboard callbacks
        self._register_callbacks()

        # track pressed keys
        self._pressed_keys = set()

        # reset environment callback
        self._reset_env_callback = None

        # create key bindings
        self._create_key_bindings()

    def __del__(self):
        """Restore the original keyboard callback."""
        if hasattr(self, "_viewer") and hasattr(self, "_original_key_callback"):
            try:
                window = self._viewer.window
                if self._original_key_callback:
                    glfw.set_key_callback(window, self._original_key_callback)
            except (AttributeError, TypeError):
                pass

    def __str__(self) -> str:
        """Returns: A string containing the information of keyboard controller."""
        msg = f"Keyboard Controller for SE(3): {self.__class__.__name__}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tMove T1 along x-axis: W/S\n"
        msg += "\tMove T1 along y-axis: A/D\n"
        msg += "\tRotate T1 along z-axis: Q/E\n"
        msg += "\tReset commands: L\n"
        msg += "\tQuit: ESC\n"      
        msg += "\tReset environment: P"
        return msg

    """
    Operations
    """

    def should_quit(self) -> bool:
        """Return True if ESC was pressed."""
        return self._should_quit

    def reset(self):
        """Reset all command buffers to default values."""
        # default flags
        self._delta_vel = np.zeros(3)  # (x, y, yaw)
        self._should_quit = False

    def advance(self) -> np.ndarray:
        """Provides the result from keyboard event state.

        Returns:
            A 3-element array containing:
            - Elements 0-1: delta position [x, y]
            - Elements 2: delta rotation [yaw]
        """
        # return the command and gripper state
        return self._delta_vel

    def add_callback(self, key: str, func: Callable):
        """Add additional functions to bind keyboard.

        Args:
            key: The keyboard button to check against.
            func: The function to call when key is pressed. The callback function should not
                take any arguments.
        """
        self._additional_callbacks[key] = func

    def set_reset_env_callback(self, callback: Callable):
        """Set the callback function to reset the environment.

        Args:
            callback: The function to call when the P key is pressed.
        """
        self._reset_env_callback = callback

    """
    Internal helpers.
    """

    def _register_callbacks(self):
        """Register GLFW keyboard callbacks."""
        # Get the GLFW window from the viewer
        window = self._viewer.window

        # Set our key callback
        glfw.set_key_callback(window, self._on_keyboard_event)

    def _on_keyboard_event(self, window, key, scancode, action, mods):
        """GLFW keyboard callback function."""
        # Convert GLFW key to character
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            self._should_quit = True
            # Optionally also request the window to close:
            try:
                glfw.set_window_should_close(window, True)
            except Exception:
                pass
            return

        try:
            # Map arrow keys directly using a dictionary
            arrow_keys = {
                glfw.KEY_LEFT: "LEFT",
                glfw.KEY_RIGHT: "RIGHT",
                glfw.KEY_UP: "UP",
                glfw.KEY_DOWN: "DOWN",
            }

            if key in arrow_keys:
                key_char = arrow_keys[key]
            else:
                key_char = chr(key).upper()
        except ValueError:
            # Not a printable character
            return

        if key_char in self._INPUT_KEY_MAPPING.keys():
            # Handle key press
            if action == glfw.PRESS:
                self._pressed_keys.add(key_char)
                self._handle_key_press(key_char)

            # Handle key release
            elif action == glfw.RELEASE:
                self._pressed_keys.discard(key_char)
                self._handle_key_release(key_char)
        else:
            if self._original_key_callback:
                self._original_key_callback(window, key, scancode, action, mods)

    def _handle_key_press(self, key_char):
        """Handle key press events."""
        # Apply the command when pressed
        if key_char == "L":
            self.reset()
        elif key_char == "P" and self._reset_env_callback:
            self._reset_env_callback()
        elif key_char in ["W", "S", "A", "D", "Q", "E"]:
            self._delta_vel += self._INPUT_KEY_MAPPING[key_char]

        # Additional callbacks
        if key_char in self._additional_callbacks:
            self._additional_callbacks[key_char]()

    def _handle_key_release(self, key_char):
        """Handle key release events."""
        # Remove the command when un-pressed
        if key_char in ["W", "S", "A", "D", "Q", "E"]:
            self._delta_vel -= self._INPUT_KEY_MAPPING[key_char]

    def _create_key_bindings(self):
        """Creates default key binding."""
        self._INPUT_KEY_MAPPING = {
            # x-axis (forward)
            "W": np.asarray([1.0, 0.0, 0.0]) * self.pos_sensitivity,
            "S": np.asarray([-1.0, 0.0, 0.0]) * self.pos_sensitivity,
            # y-axis (left-right)
            "A": np.asarray([0.0, 1.0, 0.0]) * self.pos_sensitivity,
            "D": np.asarray([0.0, -1.0, 0.0]) * self.pos_sensitivity,
            # z-axis (rotation)
            "Q": np.asarray([0.0, 0.0, 1.0]) * self.rot_sensitivity,
            "E": np.asarray([0.0, 0.0, -1.0]) * self.rot_sensitivity,
            # reset commands
            "L": self.reset,
            "P": self._reset_env_callback,
        }


class Se3Keyboard_Pynput(Se3Keyboard):
    """Keyboard controller using pynput listener with the same API as Se3Keyboard."""

    def __init__(
        self,
        renderer,
        pos_sensitivity: float,
        rot_sensitivity: float,
    ):
        if pynput_keyboard is None:
            raise ImportError(
                "pynput is required for Se3Keyboard_Pynput. Please install it with 'pip install pynput'."
            )

        self.renderer = renderer

        self._delta_vel = np.zeros(3)
        self.rot_sensitivity = rot_sensitivity
        self.pos_sensitivity = pos_sensitivity
        self._additional_callbacks: dict[str, Callable] = dict()
        self._pressed_keys: set[str] = set()
        self._reset_env_callback: Callable | None = None
        self._lock = threading.Lock()
        self._pending_reset = False
        self._pending_p_additional: Callable | None = None
        self._should_quit = False

        self._create_key_bindings()

        self._listener = pynput_keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release, suppress=False
        )
        self._listener.start()

        self.viewer = self.renderer._get_viewer("human")

    def _handle_key_press(self, key_char):
        """Handle key press events."""
        # Apply the command when pressed
        if key_char == "L":
            self.reset()
        elif key_char == "P" and self._reset_env_callback:
            self._reset_env_callback()
        elif key_char in ["V", "B", "N", "M", ".", ","]:
            self._delta_vel += self._INPUT_KEY_MAPPING[key_char]

        # Additional callbacks
        if key_char in self._additional_callbacks:
            self._additional_callbacks[key_char]()

    def _handle_key_release(self, key_char):
        """Handle key release events."""
        # Remove the command when un-pressed
        if key_char in ["V", "B", "N", "M", ".", ","]:
            self._delta_vel -= self._INPUT_KEY_MAPPING[key_char]

    def _create_key_bindings(self):
        """Creates default key binding."""
        self._INPUT_KEY_MAPPING = {
            # x-axis (forward)
            "V": np.asarray([1.0, 0.0, 0.0]) * self.pos_sensitivity,
            "B": np.asarray([-1.0, 0.0, 0.0]) * self.pos_sensitivity,
            # y-axis (left-right)
            "N": np.asarray([0.0, 1.0, 0.0]) * self.pos_sensitivity,
            "M": np.asarray([0.0, -1.0, 0.0]) * self.pos_sensitivity,
            # z-axis (rotation)
            ".": np.asarray([0.0, 0.0, 1.0]) * self.rot_sensitivity,
            ",": np.asarray([0.0, 0.0, -1.0]) * self.rot_sensitivity,
            # reset commands
            "L": self.reset,
            "P": self._reset_env_callback,
        }

    def __del__(self):
        try:
            if hasattr(self, "_listener") and self._listener is not None:
                self._listener.stop()
        except Exception:
            pass

    def set_reset_env_callback(self, callback: Callable):
        self._reset_env_callback = callback
        self._create_key_bindings()

    def _key_to_char(self, key) -> str | None:
        try:
            if isinstance(key, pynput_keyboard.KeyCode) and key.char:
                return key.char.upper()
        except Exception:
            return None
        return None

    def _on_press(self, key):
        key_char = self._key_to_char(key)
        if not key_char:
            return
        if key_char == "P":
            with self._lock:
                if key_char not in self._pressed_keys:
                    self._pressed_keys.add(key_char)
                    self._pending_reset = True
                    self._pending_p_additional = self._additional_callbacks.get("P")
            return

        if key_char in self._INPUT_KEY_MAPPING or key_char == "L":
            with self._lock:
                if key_char not in self._pressed_keys:
                    self._pressed_keys.add(key_char)
                    self._handle_key_press(key_char)

    def _on_release(self, key):
        key_char = self._key_to_char(key)
        if not key_char:
            return
        if key_char in ["V", "B", "N", "M", ".", ","]:
            with self._lock:
                if key_char in self._pressed_keys:
                    self._pressed_keys.discard(key_char)
                    self._handle_key_release(key_char)
        elif key_char == "P":
            with self._lock:
                self._pressed_keys.discard(key_char)

    def advance(self) -> np.ndarray:
        do_reset = False
        reset_cb: Callable | None = None
        p_additional: Callable | None = None
        with self._lock:
            if self._pending_reset:
                self._pending_reset = False
                do_reset = True
                reset_cb = self._reset_env_callback
                p_additional = self._pending_p_additional
                self._pending_p_additional = None
        if do_reset:
            try:
                if reset_cb is not None:
                    reset_cb()
            finally:
                if p_additional is not None:
                    try:
                        p_additional()
                    except Exception:
                        pass
        return super().advance()
    
    def __str__(self) -> str:
        """Returns: A string containing the information of keyboard controller."""
        msg = f"Keyboard Controller for SE(3): {self.__class__.__name__}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tMove T1 along x-axis: V/B\n"
        msg += "\tMove T1 along y-axis: N/M\n"
        msg += "\tRotate T1 along z-axis: ./,\n"
        msg += "\tReset commands: L\n"
        msg += "\tQuit: ESC\n"      
        msg += "\tReset environment: P"
        return msg

