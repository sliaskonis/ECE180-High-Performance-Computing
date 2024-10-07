import os
import threading
import subprocess
from decimal import Decimal
from subprocess import Popen, PIPE, CalledProcessError
import pandas as pd
import numpy as np

#stats=[0]*10
rows, cols = 10, 10
stats = None

def run_script(col, row):
    #subprocess.call(["icx", "-Wall", "-O0", script_name, "-o", "Lab1"]) 
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
    
def export_to_excel(data, file_name, cols):
    # Convert the list of numbers into a pandas DataFrame
    column_names = [str(i + 1) for i in range(cols)]
    df = pd.DataFrame(data, columns=column_names)
    df['Average'] = df.mean(axis=1)
    df['StdDev'] = df.std(axis=1)
    # Export the DataFrame to an Excel file
    df.to_excel(file_name, index=False)

if __name__ == "__main__":
    loops = int(input("number of loops: "))
    
    rows = 9
    stats = [[0.0 for _ in range(loops)] for _ in range(rows)]
    #stats=[0]*10
    subprocess.call(["icx", "-Wall", "-O0", "1.loop_interchange_1/sobel_loop_interchange_1.c", "-o", "Lab1"]) 
    for num in range(loops):
        thread = threading.Thread(target=run_script, args=(num,0))
        thread.start()
        thread.join()
    print("Done1.\n")

    subprocess.call(["icx", "-Wall", "-O0", "2.loop_interchange_2/sobel_loop_interchange_2.c", "-o", "Lab1"]) 
    for num in range(loops):
        thread = threading.Thread(target=run_script, args=(num,1))
        thread.start()
        thread.join()
    print("Done2.\n")

    subprocess.call(["icx", "-Wall", "-O0", "3.loop_unroll_conv2d/sobel_loop_unroll_conv2d.c", "-o", "Lab1"])
    for num in range(loops): 
        thread = threading.Thread(target=run_script, args=(num,2))
        thread.start()
        thread.join()
    print("Done3.\n")

    subprocess.call(["icx", "-Wall", "-O0", "4.loop_fusion/sobel_loop_fusion.c", "-o", "Lab1"]) 
    for num in range(loops):
        thread = threading.Thread(target=run_script, args=(num,3))
        thread.start()
        thread.join()
    print("Done4.\n")

    subprocess.call(["icx", "-Wall", "-O0", "5.common_sub_expr_elim/sobel_csee.c", "-o", "Lab1"]) 
    for num in range(loops):
        thread = threading.Thread(target=run_script, args=(num,4))
        thread.start()
        thread.join()
    print("Done5.\n")

    subprocess.call(["icx", "-Wall", "-O0", "6.common_sub_expr_elim_conv2d/sobel_csee_conv2d.c", "-o", "Lab1"]) 
    for num in range(loops):
        thread = threading.Thread(target=run_script, args=(num,5))
        thread.start()
        thread.join()
    print("Done6.\n")

    subprocess.call(["icx", "-Wall", "-O0", "7.function_inlining/function_inlining.c", "-o", "Lab1"]) 
    for num in range(loops):
        thread = threading.Thread(target=run_script, args=(num,6))
        thread.start()
        thread.join()
    print("Done7.\n")

    subprocess.call(["icx", "-Wall", "-O0", "8.function_inlining_without_6_change/function_inlining.c", "-o", "Lab1"]) 
    for num in range(loops):
        thread = threading.Thread(target=run_script, args=(num,7))
        thread.start()
        thread.join()
    print("Done8.\n")

    file_name = "stats.xlsx"

    # Export the numbers to Excel
    export_to_excel(stats, file_name, loops)
    
    print(f"Data exported to {file_name}")


