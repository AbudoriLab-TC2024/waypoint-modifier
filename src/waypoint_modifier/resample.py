import numpy as np
import pandas as pd
import argparse
import os


class Resampler:
    def __init__(
        self,
        x_col,
        y_col,
        curvature_col,
        distance_col,
        window,
        min_r,
        min_d,
        max_d,
        max_c,
    ):
        self._x_col = x_col
        self._y_col = y_col
        self._curvature_col = curvature_col
        self._distance_col = distance_col
        self._window = window
        self._min_r = min_r
        self._min_d = min_d
        self._max_d = max_d
        self._max_c = max_c

    def set_curvature(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the curvatures of a trajectory.
        """
        ret = []
        for i in range(len(df)):
            if i < self._window or i >= len(df) - self._window:
                ret.append(0.0)
            else:
                x = df[self._x_col].values[i - self._window : i + self._window + 1]
                y = df[self._y_col].values[i - self._window : i + self._window + 1]

                A = np.vstack((x, y, np.ones(len(x)))).T
                v = -(x**2 + y**2)
                u, *_ = np.linalg.lstsq(A, v, rcond=None)

                cx_pred = u[0] / -2
                cy_pred = u[1] / -2
                r_pred = np.sqrt(cx_pred**2 + cy_pred**2 - u[2])
                r_pred = max(r_pred, self._min_r)
                ret.append(1.0 / r_pred)

        df[self._curvature_col] = pd.Series(ret, index=df.index)

        return df

    def set_distance(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self._distance_col] = np.sqrt(
            df[self._x_col].diff() ** 2 + df[self._y_col].diff() ** 2
        ).fillna(0)
        return df

    def do_sampling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sample waypoints based on distance.
        """
        dist = 0.0
        curv = 0.0

        ret = [df.iloc[0]]
        for i in range(len(df)):
            dist += df[self._distance_col].values[i]
            curv += df[self._curvature_col].values[i]

            if (dist > self._max_d or curv > self._max_c) and dist > self._min_d:
                ret.append(df.iloc[i])
                dist = 0.0
                curv = 0.0

        ret_df = pd.DataFrame(ret).copy()
        ret_df.reset_index(drop=True, inplace=True)
        self.set_distance(ret_df)

        return ret_df

    def resample(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.set_curvature(df)
        df = self.set_distance(df)
        return self.do_sampling(df)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input CSV file")
    parser.add_argument("--output", "-o", type=str, help="Output CSV file")
    parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="Curvature computation window size (default: 20)",
    )
    parser.add_argument(
        "--min_radius",
        type=float,
        default=0.1,
        help="Minimum radius of curvature computation [m] (default: 0.1)",
    )
    parser.add_argument(
        "--min_accum_dist",
        type=float,
        default=0.2,
        help="Minimum accumulation distance between points when resampling [m] (default: 0.2)",
    )
    parser.add_argument(
        "--max_accum_dist",
        type=float,
        default=2,
        help="Maximum accumulation distance between points when resampling [m] (default: 2)",
    )
    parser.add_argument(
        "--max_accum_curv",
        type=float,
        default=3,
        help="Maximum accumulation curvature between points when resampling [1/m] (default: 3)",
    )
    parser.add_argument(
        "--simple-xy-col",
        action="store_true",
        help="Use 'x', 'y' for the input csv columns name instead of '/pcl_pose/pose/pose/positon/x,y'",
    )

    args = parser.parse_args()

    x_col, y_col = (
        ("x", "y")
        if args.simple_xy_col
        else ("/pcl_pose/pose/pose/position/x", "/pcl_pose/pose/pose/position/y")
    )

    src_df = pd.read_csv(args.input)
    resampler = Resampler(
        x_col=x_col,
        y_col=y_col,
        curvature_col="curvature",
        distance_col="distance",
        window=args.window,
        min_r=args.min_radius,
        min_d=args.min_accum_dist,
        max_d=args.max_accum_dist,
        max_c=args.max_accum_curv,
    )

    dst_df = resampler.resample(src_df)

    # rename columns to simplify the output
    dst_df = dst_df.rename(columns={x_col: "x", y_col: "y"})

    dst_filename = (
        args.output if args.output else os.path.splitext(args.input)[0] + "_sparse.csv"
    )
    if "action" not in dst_df.columns:
        dst_df["action"] = ""

    dst_df[["x", "y", "action"]].to_csv(dst_filename)


if __name__ == "__main__":
    main()
