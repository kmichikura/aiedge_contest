#ifndef AXIS_H
#define AXIS_H

#define NUM_AXIS (2)
#define AXIS_SIZE (0x00001000)
#define DDR_USIZE (0x40000000)

#define DDR_ADDR (0x40000000)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <signal.h>
#include <time.h>
#define _GNU_SOURCE
#include <poll.h>

int ppoll(struct pollfd *fds, nfds_t nfds, 
        const struct timespec *timeout_ts, const sigset_t *sigmask);



int fd_axis [NUM_AXIS];
volatile void* axis_ptr [NUM_AXIS];

void uio_name(char* uio, int i)
{
  char buf [128];
  sprintf(buf, "%d", i);
  strcat(uio, buf);
}

void axis_open()
{
  for(int i=0; i<NUM_AXIS; i++){
    char uio [1024] = "/dev/uio";
    unsigned int uio_size;
    uio_name(uio, i+1);
    fd_axis[i] = open(uio, O_RDWR);
    if(fd_axis[i] < 1){
      printf("Invalid UIO device file: '%s'\n", uio);
      exit(1);
    }
    switch (i){
      case 0 : uio_size = DDR_USIZE; break;  // DDR
      case 1 : uio_size = AXIS_SIZE; break;  // PL
      default: uio_size = AXIS_SIZE; break;
    }
    axis_ptr[i] = (volatile void*) mmap(NULL, uio_size, PROT_READ|PROT_WRITE, MAP_SHARED, fd_axis[i], 0);
    if( axis_ptr[i] == NULL ) {
      puts("NULL");
    }
  }
}

template <typename T>
void axis_write(int id, int offset, T data)
{
  *(volatile T*)(axis_ptr[id] + offset) = data;
  msync((void*)(axis_ptr[id] + offset), sizeof(T), MS_SYNC);
}

template <typename T>
void axis_read(int id, int offset, T* data)
{
  volatile T r = *(volatile T*)(axis_ptr[id] + offset);
  msync((void*)(axis_ptr[id] + offset), sizeof(T), MS_SYNC);
  *data = r;
}

void axis_write_2b(unsigned int id, unsigned int offset, short data)
{
  *(volatile short*)(axis_ptr[id] + offset) = data;
  msync((void*)(axis_ptr[id] + offset), sizeof(short), MS_SYNC);
}

void axis_read_2b(unsigned int id, unsigned int offset, short* data)
{
  volatile short r = *(volatile short*)(axis_ptr[id] + offset);
  msync((void*)(axis_ptr[id] + offset), sizeof(short), MS_SYNC);
  *data = r;
}

void axis_write_4b(unsigned int id, unsigned int offset, int data)
{
  *(volatile int*)(axis_ptr[id] + offset) = data;
  msync((void*)(axis_ptr[id] + offset), sizeof(int), MS_SYNC);
}

void axis_read_4b(unsigned int id, unsigned int offset, int* data)
{
  volatile int r = *(volatile int*)(axis_ptr[id] + offset);
  msync((void*)(axis_ptr[id] + offset), sizeof(int), MS_SYNC);
  *data = r;
}

void axis_write_8b(unsigned int id, unsigned int offset, unsigned long long data)
{
  *(volatile unsigned long long*)(axis_ptr[id] + offset) = data;
  msync((void*)(axis_ptr[id] + offset), sizeof(unsigned long long), MS_SYNC);
}

void axis_read_8b(unsigned int id, unsigned int offset, unsigned long long* data)
{
  volatile unsigned long long r = *(volatile unsigned long long*)(axis_ptr[id] + offset);
  msync((void*)(axis_ptr[id] + offset), sizeof(unsigned long long), MS_SYNC);
  *data = r;
}

void axis_irq_on(unsigned int id){
  unsigned int irq_on = 1;
  write(fd_axis[id], &irq_on, sizeof(irq_on));
}

void axis_irq_off(unsigned int id){
  unsigned int irq_off = 0;
  write(fd_axis[id], &irq_off, sizeof(irq_off));
}

void axis_irq_wait(unsigned int id){
  struct pollfd   fds[1];
  struct timespec timeout;
  sigset_t        sigmask;
  int             poll_result;
  unsigned int    irq_count;
  fds[0].fd       = fd_axis[id];
  fds[0].events   = POLLIN;
  timeout.tv_sec  = 5;
  timeout.tv_nsec = 0;
  poll_result = ppoll(fds, 1, &timeout, &sigmask);
  if ((poll_result > 0) && (fds[0].revents & POLLIN)) {
    read(fd_axis[id], &irq_count,  sizeof(irq_count));
  }
  else if(poll_result == -1){
    printf("uio_irq error\n");
  }
  else if(poll_result == 0){
    printf("uio_irq time out!\n");
  }
}

void axis_irq_wait_sec(unsigned int id, unsigned int sec){
  struct pollfd   fds[1];
  struct timespec timeout;
  sigset_t        sigmask;
  int             poll_result;
  unsigned int    irq_count;
  fds[0].fd       = fd_axis[id];
  fds[0].events   = POLLIN;
  timeout.tv_sec  = sec;
  timeout.tv_nsec = 0;
  poll_result = ppoll(fds, 1, &timeout, &sigmask);
  if ((poll_result > 0) && (fds[0].revents & POLLIN)) {
    read(fd_axis[id], &irq_count,  sizeof(irq_count));
  }
  else if(poll_result == -1){
    printf("uio_irq error\n");
  }
  else if(poll_result == 0){
    printf("uio_irq time out!\n");
  }
}

void axis_close()
{
  for(int i=0; i<NUM_AXIS; i++){
    unsigned int uio_size;
    switch (i) {
      case 0 : uio_size = DDR_USIZE; break;
      case 1 : uio_size = AXIS_SIZE; break;
      default: uio_size = AXIS_SIZE; break;
    }
    munmap((void*) axis_ptr[i], uio_size);
    axis_ptr[i] = NULL;
    close(fd_axis[i]);
    fd_axis[i] = -1;
  }
}

#endif
