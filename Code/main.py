import numpy;
import torch;

from Settings_Reader import Settings_Reader, Settings_Container;
from Mappings import xy_Derivatives_to_Index, Index_to_xy_Derivatives_Class, \
                     Num_Multi_Indices, Multi_Indices_Array, \
                     Multi_Index_to_Col_Number_Class, Col_Number_to_Multi_Index_Class;


def main():
    Settings = Settings_Reader();

    for (Setting, Value) in Settings.__dict__.items():
        print(("%-25s = " % Setting) + str(Value));


    # 2D Case.

    # Set up Index_to_xy_Derivatives, Multi_Index_to_Col_Number, and
    # Col_Number_to_Multi_Index maps.
    Index_to_xy_Derivatives = Index_to_xy_Derivatives_Class(
                                    Highest_Order_Derivatives   = Settings.Highest_Order_Derivatives);
    Multi_Index_to_Col_Number = Multi_Index_to_Col_Number_Class(
                                    Max_Sub_Indices             = Settings.Maximum_Term_Degree,
                                    Num_Sub_Index_Values        = Index_to_xy_Derivatives.Num_Index_Values);
    Multi_Index_to_Col_Number = Col_Number_Tt_Multi_Index_Class(
                                    Max_Sub_Indices             = Settings.Maximum_Term_Degree,
                                    Num_Sub_Index_Values        = Index_to_xy_Derivatives.Num_Index_Values);





if(__name__ == "__main__"):
    main();
