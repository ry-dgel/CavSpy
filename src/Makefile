CC = gcc  # C compiler

ifeq ($(OS),Windows_NT)
    DLLEXT := .dll
	CFLAGS = -fPIC -fopenmp -fstack-protector -flto -Wall -DBUILD_DLL -Wextra -O2 -g# C flags
	LDFLAGS = -shared -lm -lgomp -flto -fopenmp -fstack-protector# linking flags
	RM = del /f   # rm command
else
    DLLEXT := .so
	CFLAGS = -fPIC -fopenmp -Wall -Wextra -O2 -g  # C flags
	LDFLAGS = -shared -lm -fopenmp # linking flags
	RM = rm -f   # rm command
endif

DLL := minPar$(DLLEXT)
TARGET_LIB = ../$(DLL)  # target lib

SRCS = minPar.c  # source files
OBJS = $(SRCS:.c=.o)

.PHONY: all
all: ${TARGET_LIB}

$(TARGET_LIB): $(OBJS)
	$(CC) ${LDFLAGS} -o $@ $^

$(SRCS:.c=.d):%.d:%.c
	$(CC) $(CFLAGS) -MM $< >$@

include $(SRCS:.c=.d)

.PHONY: clean
clean:
	-${RM} ${OBJS} $(SRCS:.c=.d)