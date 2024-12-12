from contextlib import closing
import os
import sys
import platform
import socket
import subprocess
from subprocess import PIPE
import tempfile
import time

from py4j.java_gateway import JavaGateway, GatewayParameters
from py4j.java_collections import ListConverter

import urllib.request
from pathlib import Path

class ErodeHandler:
    ## Specify path to java JDK
    #Default string. Can be changed using the constructor of erodeHandler
    #__JAVA_PATH__="/Library/Java/JavaVirtualMachines/jdk-11.0.14.jdk/Contents/Home/bin/java"
    #__JAVA_PATH__="/usr/lib/jvm/java-8-openjdk-amd64/Contents/Home/bin/java"
    __JAVA_PATH__="java"
    __IN_COLAB__ = False #'google.colab' in sys.modules

    ## Specify path to java JDK
    #Default string.
    __ERODE_JAR__ = "erode.jar" #os.path.join(os.path.dirname(__file__), "erode.jar")

    ## Specify whether a new JVM shall be created, or whether we shall connect to an existing one
    # Do not change
    __STARTJVM__ = True

    def __init__(self,j_path=__JAVA_PATH__):
        self._java_path=j_path
        self.erode_jar=ErodeHandler.__ERODE_JAR__

    def _start_server(self):
        # find a free port
        for port in range(25333, 65545):
            with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
                dest_addr = ("127.0.0.1", port)
                if s.connect_ex(dest_addr):
                    break

        #ld_path = __ERODE_LIB_DIR__
        ld_path='native/osx'
        #java="java"
        java =self._java_path
        argv = [java, f'-Djava.library.path="{ld_path}"',
                    "-jar", self.erode_jar, str(port)]
        #print("argv",argv)
        if platform.system() == "Linux":
            env_ld_path = os.getenv("LD_LIBRARY_PATH")
            if env_ld_path:
                ld_path = f"{ld_path}:{env_ld_path}"
            env ={"LD_LIBRARY_PATH": ld_path}
            proc = subprocess.Popen(" ".join(argv), stdout=PIPE,
                    shell=True, env=env)
        else:
            proc = subprocess.Popen(argv, stdout=PIPE)
        proc.stdout.readline()

        self._proc=proc
        self._port=port

        time.sleep(1) # Sleep for 1 second
        return #proc, port

    def _stop_server(self):
        if ErodeHandler.__STARTJVM__:
            print('Terminating JVM and ERODE')
            time.sleep(1) # Sleep for 1 second
            self._proc.terminate()
            try:
                self._proc.wait(5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
            print(' Completed')
        else:
            print('Nothing to terminate')

    def start_JVM(self):
        print('Starting the JVM and ERODE')
        if ErodeHandler.__STARTJVM__:
            #self._proc, self._port = _start_server()
            self._start_server()
        else:
            self._proc =-1
            self._port =25347
        gw_params = GatewayParameters(port=self._port)#, auto_convert=True)
        self._gw = JavaGateway(gateway_parameters=gw_params)

        self.int_class   =self._gw.jvm.int    # make int class
        self.double_class=self._gw.jvm.double # make double class
        self.erode = self._gw.entry_point
        #print(_proc)
        #_port
        print('  Completed')
        return self.erode #,self._proc,self._port

    def j_to_py_matrix(self,metrics_java):
        metrics_python= [ list(line) for line in metrics_java ]
        return metrics_python

    def j_to_py_list(self,partition_java):
        partition_python= [ int(entry) for entry in partition_java ]
        return partition_python


    #Takes in input a list of int created in python. Gives in output a corresponding array of int for java
    def py_to_j_list(self,python_int_list):
        java_int_array = self._gw.new_array(self.int_class,len(python_int_list))
        #double_array = gateway.new_array(self.double_class,1,len(partition_python))

        for i in range(len(python_int_list)):
            java_int_array[i]=python_int_list[i]

        return java_int_array

    def matrix_to_upper_diagonal(m):
        l=[]
        for i in range(len(m)):
            for j in range(i+1,len(m)):
                l.append(m[i][j])
        return l

#Always start ERODE on a JVM
erodeHandler = ErodeHandler()
erode = erodeHandler.start_JVM()
type(erode)