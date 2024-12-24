import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk
import pandas as pd
import traceback
from datetime import datetime

# Import your existing functions from main.py
from main import (
    load_and_preprocess_data,
    train_model,
    evaluate_model,
    erlang_c,
    generate_staffing_recommendations
)

class StaffingApp(Gtk.Window):
    def __init__(self):
        super().__init__(title="Staffing Recommendation Tool")
        self.set_border_width(10)
        self.set_default_size(600, 400)

        # Layout
        layout = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.add(layout)

        # File path input
        self.file_entry = Gtk.Entry()
        self.file_entry.set_placeholder_text("Enter the CSV file path...")
        layout.pack_start(self.file_entry, False, False, 0)

        # Start date input
        self.start_date_entry = Gtk.Entry()
        self.start_date_entry.set_placeholder_text("Enter start date (YYYY-MM-DD)...")
        layout.pack_start(self.start_date_entry, False, False, 0)

        # End date input
        self.end_date_entry = Gtk.Entry()
        self.end_date_entry.set_placeholder_text("Enter end date (YYYY-MM-DD)...")
        layout.pack_start(self.end_date_entry, False, False, 0)

        # Button to process data
        self.process_button = Gtk.Button(label="Generate Staffing Recommendations")
        self.process_button.connect("clicked", self.on_process_clicked)
        layout.pack_start(self.process_button, False, False, 0)

        # Text view to display results
        self.result_view = Gtk.TextView()
        self.result_view.set_editable(False)
        self.result_view.set_wrap_mode(Gtk.WrapMode.WORD)
        self.result_buffer = self.result_view.get_buffer()
        layout.pack_start(self.result_view, True, True, 0)

    def on_process_clicked(self, widget):
        file_path = self.file_entry.get_text()
        start_date = self.start_date_entry.get_text()
        end_date = self.end_date_entry.get_text()

        try:
            # Validate inputs
            if not file_path or not start_date or not end_date:
                raise ValueError("All fields must be filled.")
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
            end_date = datetime.strptime(end_date, "%Y-%m-%d")

            # Load and preprocess data
            data_encoded, original_data = load_and_preprocess_data(file_path)
            X_train, X_test, y_train, y_test = train_model(data_encoded)
            model, _ = train_model(X_train, y_train)
            y_pred = evaluate_model(model, X_test, y_test)

            # Generate future predictions
            future_dates = pd.date_range(start=start_date, end=end_date)
            future_predictions = pd.DataFrame({'date': future_dates, 'calls_offered_predicted': y_pred[:len(future_dates)]})
            staffed_predictions = generate_staffing_recommendations(future_predictions)

            # Display results
            result_text = staffed_predictions.to_string(index=False)
            self.result_buffer.set_text(result_text)

        except Exception as e:
            # Display error messages
            error_message = f"Error: {str(e)}\n\n{traceback.format_exc()}"
            self.result_buffer.set_text(error_message)

# Run the GTK application
if __name__ == "__main__":
    app = StaffingApp()
    app.connect("destroy", Gtk.main_quit)
    app.show_all()
    Gtk.main()
