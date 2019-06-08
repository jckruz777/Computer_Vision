import pickle

# Dictionary with the initial values
init_values = { 'noise_kernel_dim': 3, 'closing_iterations': 2, 'background_iterations': 3, 'thresh_factor': 0.2, 'morph_selector': 0, 'hue_labels': 70}

# Serializing
filename = 'watershed'
outfile = open(filename,'wb')
pickle.dump(init_values, outfile)
outfile.close()
