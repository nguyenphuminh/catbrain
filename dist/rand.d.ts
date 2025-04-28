export declare const weightInitWithAct: Record<string, string>;
export declare class Rand {
    static uniform(start: number, end: number): number;
    static normal(mean?: number, stdDev?: number): number;
    static xavierUniform(inSize: number, outSize: number): number;
    static xavierNormal(inSize: number, outSize: number): number;
    static heUniform(inSize: number): number;
    static heNormal(inSize: number): number;
    static lecunUniform(inSize: number): number;
    static lecunNormal(inSize: number): number;
    static basicUniform(): number;
}
