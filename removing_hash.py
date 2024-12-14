gal = [str(i).zfill(3) for i in range(100)]
obj = [str(i) for i in range(100)]
        
for i in range(len(gal)):
    for j in range(len(obj)):
        old_file = '/Users/sray/Documents/Saavik_Barry/test_mcfacts/runs/gal' + gal[i] + '/output_bh_binary_' + obj[j] + '.dat'
        with open(old_file, 'r') as files:
            lines = files.readlines()

        if lines:
            lines[0] = lines[0][2:]

        new_file = '/Users/sray/Documents/Saavik_Barry/test_mcfacts/runs/gal' + gal[i] + '/output_bh_binary_' + obj[j] + '.dat'
        with open(new_file, 'w') as files:
            files.writelines(lines)