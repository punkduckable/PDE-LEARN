import numpy;
import torch;

from Settings_Reader import Settings_Reader, Settings_Container;

def main():
    Settings = Settings_Reader();

    for (Setting, Value) in Settings.__dict__.items():
        print(("%-25s = " % Setting) + str(Value));

if(__name__ == "__main__"):
    main();
