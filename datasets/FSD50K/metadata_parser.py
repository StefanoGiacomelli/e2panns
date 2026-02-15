import csv

# Define positive and negative labels
positives = ['Siren']
negatives = ['Car', 'Car_passing_by', 'Race_car_and_auto_racing', 'Vehicle_horn_and_car_horn_and_honking',
             'Accelerating_and_revving_and_vroom', 'Bus', 'Motor_vehicle_(road)', 'Motorcycle', 'Truck', 'Bicycle',
             'Rail_transport', 'Train', 'Subway_and_metro_and_underground', 'Skateboard', 'Traffic_noise_and_roadway_noise',
             'Bicycle_bell', 'Alarm', 'Telephone', 'Doorbell', 'Ringtone', 'Speech', 'Music', 'Engine', 'Engine_starting', 'Idling']

# Input CSV
#input_csv = "./datasets/FSD50K/FSD50K.ground_truth/dev.csv"
input_csv = "./datasets/FSD50K/FSD50K.ground_truth/eval.csv"

# Output CSVs
#output_positive_csv = "./FSD-dev_positives.csv"
#output_negative_csv = "./FSD-dev_negatives.csv"
output_positive_csv = "./FSD-eval_positives.csv"
output_negative_csv = "./FSD-eval_negatives.csv"

def filter_csv(input_file, positive_file, negative_file):
    with open(input_file, mode='r', newline='', encoding='utf-8') as infile, open(positive_file, mode='w', newline='', encoding='utf-8') as pos_outfile, open(negative_file, mode='w', newline='', encoding='utf-8') as neg_outfile: 
        reader = csv.reader(infile)
        pos_writer = csv.writer(pos_outfile)
        neg_writer = csv.writer(neg_outfile)
        
        # Read header and write to both output files
        header = next(reader)
        pos_writer.writerow(header)
        neg_writer.writerow(header)
        
        for row in reader:
            labels = row[1].split(',')  # Extract labels from the second column
            
            if any(label in positives for label in labels):
                pos_writer.writerow(row)  # Write to positives CSV
            elif any(label in negatives for label in labels) and not any(label in positives for label in labels):
                neg_writer.writerow(row)  # Write to negatives CSV

# Run the filtering function
filter_csv(input_csv, output_positive_csv, output_negative_csv)

# Count the number of rows in each CSV
with open(output_positive_csv, mode='r', newline='', encoding='utf-8') as pos_file, open(output_negative_csv, mode='r', newline='', encoding='utf-8') as neg_file:
    pos_count = sum(1 for row in csv.reader(pos_file))
    neg_count = sum(1 for row in csv.reader(neg_file))
    print(f"Number of positive samples: {pos_count}")
    print(f"Number of negative samples: {neg_count}")