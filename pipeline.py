import argparse
from commands import reset, train, sort

def main():
    parser = argparse.ArgumentParser(description = 'pipeline.py commands')
    
    # Define subparsers for each command
    subparsers = parser.add_subparsers(dest = 'command')

    train_command = subparsers.add_parser(name = 'train', help = 'Use train to train the model. Follow by --epochs=n_epochs \
                                          (-e=n_epochs) and --save_interval=save_interval (-s=save_interval).')
    train_command.add_argument('-e', '--epochs', type = int, help = 'Number of epochs you want to train the model.')
    train_command.add_argument('-s', '--save_interval', type = int, help = 'Define how often you want to save the model. Each time the current epoch is a multiple of the interval, \
                               the model is saved to disk. Default is None, so the model only is saved after finishing all epochs.')
    train_command.add_argument('-p', '--push', action = 'store_true', help = 'Pushes changes to github automatically on each save if selected.')
    
    reset_command = subparsers.add_parser(name = 'reset', help = 'Use reset to reset the current model. Chose --hard for hard reset and delete the current model and its history.')
    reset_command.add_argument('--hard', action = 'store_true')

    sort_command = subparsers.add_parser(name = 'sort', help = 'Sorts a newly imported car_make_images folder so it can be yielded by flow_from_directory method.')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Dispatch to the appropriate function based on the command
    if args.command == 'train':
        train.run(args.epochs, args.save_interval, args.push)
    elif args.command == 'reset':
        reset.run(args.hard)
    elif args.command == 'sort':
        sort.run()
    else:
        print('Invalid command. Use --help for usage information.')


if __name__ == '__main__':
    main()