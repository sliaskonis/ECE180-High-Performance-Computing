import os
import threading
import subprocess
from decimal import Decimal
from subprocess import Popen, PIPE, CalledProcessError
import pandas as pd
import numpy as np

#stats=[0]*10
stats = np.array([[0., 0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0., 0.]])

def run_script(script_name, col, row):
    subprocess.call(["icx", "-Wall", "-O0", script_name, "-o", "Lab1"]) 
    #tmp=Decimal(subprocess.call("./Lab1"))
    #print(tmp)
    with Popen("./Lab1", shell=True, stdout=PIPE, bufsize=1, universal_newlines=True) as p:
        for line in p.stdout:
            if line[0:10] == "Total time":
                print(line[16:23]) # process line here
                stats[row][col] = float(line[16:23])
                #stats[row][col] = stats[row][col] +float(line[16:23])
                #print(stats[indx])
                break
    if p.returncode != 0:
        raise CalledProcessError(p.returncode, p.args)
    
def export_to_excel(data, file_name):
    # Convert the list of numbers into a pandas DataFrame
    df = pd.DataFrame(data, columns=['A', 'B', 'C', 'D', 'E', 'F'])
    
    # Export the DataFrame to an Excel file
    df.to_excel(file_name, index=False)


if __name__ == "__main__":
    loops = int(input("number of loops: "))

    #stats=[0]*10
    for num in range(loops):
        thread = threading.Thread(target=run_script, args=("1.loop_interchange_1/sobel_loop_interchange_1.c",num,0))
        thread.start()
        thread.join()
    print("Done1.\n")
    for num in range(loops):
        thread = threading.Thread(target=run_script, args=("2.loop_interchange_2/sobel_loop_interchange_2.c",num,1))
        thread.start()
        thread.join()
    print("Done2.\n")
    for num in range(loops):
        thread = threading.Thread(target=run_script, args=("3.loop_unroll_conv2d/sobel_loop_unroll_conv2d.c",num,2))
        thread.start()
        thread.join()
    print("Done3.\n")
    for num in range(loops):
        thread = threading.Thread(target=run_script, args=("4.loop_fusion/sobel_loop_fusion.c",num,3))
        thread.start()
        thread.join()
    print("Done4.\n")
    for num in range(loops):
        thread = threading.Thread(target=run_script, args=("5.common_sub_expr_elim/sobel_csee.c",num,4))
        thread.start()
        thread.join()
    print("Done5.\n")
    for num in range(loops):
        thread = threading.Thread(target=run_script, args=("6.common_sub_expr_elim_conv2d/sobel_csee_conv2d.c",num,5))
        thread.start()
        thread.join()
    print("Done6.\n")

    file_name = "stats.xlsx"
    
    # Export the numbers to Excel
    export_to_excel(stats, file_name)
    
    print(f"Data exported to {file_name}")


