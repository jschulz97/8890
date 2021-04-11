############################################################
# Formatting functions
############################################################
def new_section(*text):
    div = '#' * 50
    print()
    print(div)
    for t in text:
        print(t, end=' ')
    print('\n'+div, '\n')

def end_section():
    div = '-' * 30
    print()
    print(div)
    print('Section completed')
    print(div, '\n')

def note(*text):
    print('## NOTE:', end=' ')
    for t in text:
        print(t, end=' ')
    print()

debug = True

def debug(*text):
    try: 
        debug

        if(debug):
            print('## DEBUG:', end=' ')
            for t in text:
                print(t, end=' ')
            print()

    except KeyError: 
        print('Define variable "debug" before using debug printing')
        exit(0)