#============================================================================#
# Copyright 2009 Google Inc.                                                 #
#                                                                            # 
# Licensed under the Apache License, Version 2.0 (the "License");            #
# you may not use this file except in compliance with the License.           #
# You may obtain a copy of the License at                                    #
#                                                                            #
#      http://www.apache.org/licenses/LICENSE-2.0                            #
#                                                                            #
# Unless required by applicable law or agreed to in writing, software        #
# distributed under the License is distributed on an "AS IS" BASIS,          #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   #
# See the License for the specific language governing permissions and        #
# limitations under the License.                                             #
#============================================================================#

ARMADILLO_INC = /vol6/home/kd_yjp/myw/softwares/armadillo/include
ARMADILLO_LIB = /vol6/home/kd_yjp/myw/softwares/armadillo/lib64

HDF5_INC = /vol6/home/kd_yjp/myw/softwares/hdf5/include
HDF5_LIB = /vol6/home/kd_yjp/myw/softwares/hdf5/lib

INC_FLAGS = -I$(ARMADILLO_INC)
INC_FLAGS += -I$(HDF5_INC)

LD_FLAGS = -L$(ARMADILLO_LIB) -larmadillo
LD_FLAGS += -L$(HDF5_LIB) -lhdf5

GCC= g++ -std=c++11 -O3 -lm -Wall -g -gdwarf-2

sofia-kmeans:
	$(GCC) -o sofia-kmeans sofia-kmeans.cc sf-cluster-centers.cc sf-kmeans-methods.cc  ../src/sf-weight-vector.cc  ../src/sf-data-set.cc  ../src/sf-sparse-vector.cc $(INC_FLAGS) $(LD_FLAGS)
	cp sofia-kmeans ..

clean:
	rm -f ../sofia-kmeans
	rm -f sofia-kmeans
