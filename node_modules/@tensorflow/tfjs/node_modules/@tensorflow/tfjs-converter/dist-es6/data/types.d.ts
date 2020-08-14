import { DataType, Tensor } from '@tensorflow/tfjs-core';
export declare type NamedTensorMap = {
    [key: string]: Tensor;
};
export declare type NamedTensorsMap = {
    [key: string]: Tensor[];
};
export interface TensorInfo {
    name: string;
    shape?: number[];
    dtype: DataType;
}
