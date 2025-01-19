import os
import socket

if __name__ == "__main__":
    master_addr = os.environ['MASTER_ADDR']
    master_port = os.environ['MASTER_PORT']
    
    print(f"MASTER_ADDR: {master_addr}")
    print(f"MASTER_PORT: {master_port}")
    
    conn_success = False
    try:
        with socket.create_connection((master_addr, int(master_port)), timeout=5) as conn:
            print(f"Connection successful to MASTER_ADDR={master_addr} and MASTER_PORT={master_port}")
            conn_success = True
    except Exception as conn_err:
        print(f"Connection failed: {conn_err}")
        
    if not conn_success:
        master_addr = os.environ['MASTER_ADDR_FALLBACK']
        
        try:
            with socket.create_connection((master_addr, int(master_port)), timeout=5) as conn:
                print(f"Connection successful to MASTER_ADDR_FALLBACK={master_addr} and MASTER_PORT={master_port}")
        except Exception as fallback_conn_err:
            print(f"Connection failed to MASTER_ADDR_FALLBACK: {fallback_conn_err}")
            