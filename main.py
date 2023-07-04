import curses

from associate_class import *
from media_with_graph import *
from select_images_amostrarion import *

source_directory = 'destino_exemple/inferencias'
destination_directory = 'destino_exemple/selecao'
num_images_to_select = 22

directory_path = 'bboxes_yolo_exemple'
output_file = 'anotacoes.csv'
classes_file = 'classes.txt'

def main(stdscr):
    curses.curs_set(0)  # hide the cursor
    stdscr.nodelay(1)  # Makes entry non-blocking
    stdscr.keypad(1)  # Enables the use of special keys

    options = ['separation', 'statistics']
    current_option = 0

    while True:
        stdscr.clear()

        question = "Do you want to separate the samples or do the statistics?"
        option_string = " ".join([f"[{option}]" if i == current_option else option for i, option in enumerate(options)])
        stdscr.addstr(0, 0, question)
        stdscr.addstr(1, 0, option_string)

        key = stdscr.getch()

        if key == curses.KEY_RIGHT:
            current_option = (current_option + 1) % len(options)
        elif key == curses.KEY_LEFT:
            current_option = (current_option - 1) % len(options)
        elif key == ord('\n'): 
            break

    if options[current_option] == 'separation':
        select_random_images(source_directory, destination_directory, num_images_to_select)
    elif options[current_option] == 'statistics':
        process_directory(directory_path, output_file, classes_file)
        calculate_class_frequency(output_file)

    stdscr.getch()

if __name__ == '__main__':
    curses.wrapper(main)