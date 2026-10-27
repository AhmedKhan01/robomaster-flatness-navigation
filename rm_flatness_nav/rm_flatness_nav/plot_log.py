"""Generate the assignment figure and LaTeX table from a real execution log."""
import argparse
import csv
from pathlib import Path
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('execution_csv', type=Path)
    args = parser.parse_args()
    path = args.execution_csv.expanduser().resolve()
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('No execution data found')
    t = [float(r['timestamp_s']) for r in rows]
    v = [float(r['v_m_s']) for r in rows]
    voltage = [float(r['V1_V']) for r in rows]
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
    ax[0].plot(t,v,color='#176b9e',label='Applied forward command')
    ax[0].axhline(.16,color='gray',linestyle='--',label='0.16 m/s limit')
    ax[0].set_ylabel('Forward speed (m/s)')
    ax[1].plot(t,voltage,color='#a34a24',label='V1: Eq. 5 model estimate')
    ax[1].set_ylabel('Estimated voltage (V)')
    ax[1].set_xlabel('Actual elapsed time (s)')
    for a in ax:
        a.grid(alpha=.25)
        a.legend(loc='best',fontsize=8)
    fig.suptitle('RoboMaster execution: speed command and modeled motor voltage')
    fig.tight_layout()
    for extension in ('png','pdf'):
        fig.savefig(path.parent/f'speed_voltage.{extension}',dpi=180)
    plt.close(fig)
    sampled_path = path.parent/'motor_voltage_log.csv'
    with sampled_path.open() as f:
        sampled = list(csv.DictReader(f))
    with (path.parent/'telemetry_table.tex').open('w') as f:
        f.write('\\begin{longtable}{rrrrrl}\n\\toprule\n')
        f.write('$t$ (s) & $V_1$ (V) & $V_2$ (V) & $V_3$ (V) & $V_4$ (V) & State \\\\\n\\midrule\n')
        for row in sampled:
            nums = [float(row[k]) for k in ['timestamp_s','V1_V','V2_V','V3_V','V4_V']]
            state = row['state'].replace('_', '\\_')
            f.write(' & '.join(f'{n:.3f}' for n in nums)+' & '+state+' \\\\\n')
        f.write('\\bottomrule\n\\end{longtable}\n')
    arrivals = [r for r in rows if r['state'].startswith('ARRIVED')]
    print('Created speed_voltage.png, speed_voltage.pdf, telemetry_table.tex in',path.parent)
    print('Recorded arrival state:', bool(arrivals))
    print('These voltages are model estimates from applied commands, not voltmeter measurements.')
    if arrivals:
        tail = arrivals[-min(20,len(arrivals)):]
        peak = max(abs(float(r[k])) for r in tail for k in ['V1_V','V2_V','V3_V','V4_V'])
        print('Maximum absolute modeled voltage over final arrival samples:',peak)


if __name__ == '__main__':
    main()
