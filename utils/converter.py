import csv

# convert a list of numbers separated by comma into a list of letters with no spaces

def convert_to_letters(input_filename, output_filename):
    # CONVERT EVERITHING IN LETTERS "A", "C", "G", "T"

    with open(input_filename, 'r') as instance, open(output_filename, 'w') as instance_converted:
        reader = csv.reader(instance)
        

        for row in reader:
            for i in range(len(row)):
                if row[i] == '0':
                    row[i] = 'A'
                elif row[i] == '1':
                    row[i] = 'C'
                elif row[i] == '2':
                    row[i] = 'G'
                elif row[i] == '3':
                    row[i] = 'T'
                else:
                    row[i] = 'N'
            row_stringata = ''.join(row)
            instance_converted.write(row_stringata + '\n')



            
convert_to_letters('tests/test_toy.csv', 'tests/test_toy_letters.csv')