# Lab3: Edge Detection with CUDA

## Introduction

Edge Detection: Identifying points in a digital image at which the image brightness changes sharply.

Goal: Transform sequential sample code to CUDA version.

## Compile

```shell
make
```

## Run

```shell
./lab3 [INPUT_IMG] [OUTPUT_IMG]
```

e.g.

```shell
./lab3 ./testcases/jerry.png out.png
```

## Judge

```shell
png-diff [OUTPUT_IMG] [ANS_IMG]
```

e.g.

```shell
png-diff out.png ./testcases/jerry.out.png
```
