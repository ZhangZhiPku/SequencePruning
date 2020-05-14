def write_record(record_file, training_recorder, evaluation_recorder):
    with open(record_file, 'w+') as file:
        file.write('***** START OF RECORD *****\n')

        file.write('----- START OF TRAINING RECORD -----\n')
        for record_item in training_recorder:
            file.write('\tSTART OF %s\n' % record_item)
            for record_value in training_recorder[record_item]:
                file.write('\t%s\n' % record_value)
            file.write('\tEND OF %s\n' % record_item)
        file.write('----- END OF TRAINING RECORD -----\n')

        file.write('----- START OF EVALUATION RECORD -----\n')
        for record_item in evaluation_recorder:
            file.write('\tSTART OF %s\n' % record_item)
            for record_value in evaluation_recorder[record_item]:
                file.write('\t%s\n' % record_value)
            file.write('\tEND OF %s\n' % record_item)
        file.write('----- END OF EVALUATION RECORD -----\n')

        file.write('***** END OF RECORD *****\n')