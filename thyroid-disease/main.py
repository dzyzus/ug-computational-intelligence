import Algorithms.thyroid_dt as thyroid_dt
import Algorithms.thyroid_neural_network as neuron_network

while True:
    model_choice = input("Which model would you like to use?\n'D' for Decision Tree\n'N' for Neural Network\n'Q' to quit: \nCommand: ").upper()

    if model_choice == "Q":
        print("Exiting the program.")
        break
    elif model_choice == "D":
        thyroid_dt.run()
    elif model_choice == "N":
        neuron_network.run()
    else:
        print("Invalid choice. Please enter 'D', 'N', or 'Q'.")