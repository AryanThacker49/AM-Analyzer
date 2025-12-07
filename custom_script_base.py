# ==============================================================================
# AM ANALYZER - CUSTOM SCRIPT INTERFACE
# ==============================================================================
# Users should create a class that inherits from 'CustomScript'.
#
# Example:
# class MyPorosityDetector(CustomScript):
#     def setup_ui(self):
#         self.add_number_input("Threshold", default=120)
#
#     def run(self, image, metadata, params):
#         # ... logic ...
#         return output_image, results_dict
# ==============================================================================

class CustomScript:
    def __init__(self):
        self.inputs = [] # List of defined inputs to build

    def setup_ui(self):
        """
        OVERRIDE THIS.
        Define the inputs you need from the user here.
        """
        pass

    def run(self, image, metadata, params):
        """
        OVERRIDE THIS.
        The core logic of your script.
        
        Args:
            image (numpy array): The image data (BGR color).
            metadata (dict): The full metadata dictionary for this image.
                             Access scale via metadata['microns_per_pixel'].
            params (dict): The values from your UI inputs.
                           e.g. params['Threshold']
        
        Returns:
            tuple: (processed_image (numpy array), results_data (dict))
                   processed_image: The image to display on screen.
                   results_data: A flat dictionary of results to save to CSV.
                                 e.g. {'count': 5, 'area': 500.2}
        """
        raise NotImplementedError("You must implement the run() method.")

    # ==========================================================================
    # UI BUILDER FUNCTIONS (Call these in setup_ui)
    # ==========================================================================

    def add_number_input(self, label, default=0.0):
        """Adds a text box for entering a number (float or int)."""
        self.inputs.append({
            "type": "number",
            "label": label,
            "default": default
        })

    def add_text_input(self, label, default=""):
        """Adds a text box for entering a string."""
        self.inputs.append({
            "type": "text",
            "label": label,
            "default": default
        })

    def add_checkbox(self, label, default=False):
        """Adds a checkbox (True/False)."""
        self.inputs.append({
            "type": "bool",
            "label": label,
            "default": default
        })

    def add_dropdown(self, label, options, default_index=0):
        """
        Adds a dropdown menu.
        options: List of strings, e.g. ["Method A", "Method B"]
        """
        self.inputs.append({
            "type": "dropdown",
            "label": label,
            "options": options,
            "default": default_index
        })

    def add_slider(self, label, min_val, max_val, default):
        """
        Adds a slider (Integer values only).
        """
        self.inputs.append({
            "type": "slider",
            "label": label,
            "min": int(min_val),
            "max": int(max_val),
            "default": int(default)
        })