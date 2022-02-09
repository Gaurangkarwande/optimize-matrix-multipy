.PHONY: all clean

CC = gcc
INCLUDES = -I .
TARGET = matmul
MAP_FILE = matmul.map

CC_OPTIONS = -Wall -O2 -g
LD_FLAGS = -Xlinker -Map=$(MAP_FILE)
LD_LIBS = -lblas -lm

OBJ_FILES = main.o 

.SILENT: clean all %.o

all: ${OBJ_FILES}
	@echo "Building ${TARGET} ..."
	${CC} ${CC_OPTIONS} ${LD_FLAGS} ${OBJ_FILES} -o ${TARGET} ${LD_LIBS}

%.o: %.c
	@echo "Compiling $*.o ..."
	${CC} ${CC_OPTIONS} $(LD_FLAGS) -c $*.c ${INCLUDES}

clean:
	@echo "Cleaning $(OBJ_FILES) $(TARGET) $(MAP_FILE)..."
	-rm -rf	 *.o
	-rm -rf	 matmul.map
	-rm -rf	 matmul