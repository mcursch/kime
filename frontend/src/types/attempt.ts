export type TechniqueType = "front_kick" | "roundhouse_kick" | "straight_punch";

export interface Attempt {
  id: string;
  created_at: string; // ISO 8601 timestamp
  technique_type: TechniqueType;
  overall_score: number; // 0–100
}
