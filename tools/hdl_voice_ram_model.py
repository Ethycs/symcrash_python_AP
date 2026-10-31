"""Restricted 2048x9 common-clock DP16KD functional model for voice tests.

Lattice FPGA-TN-02204-1.6 section4.3/appendixA; pinned Yosys brams_map.v.
This is a checked functional profile, not a vendor timing or board model.
"""
from pathlib import Path

def validate_ram_mapping(top):
    rams=[c for c in top['cells'].values() if c['type']=='DP16KD'];assert len(rams)==15
    for cell in rams:
        p,c=cell['parameters'],cell['connections']
        for port in ('A','B'):
            assert int(p['DATA_WIDTH_'+port],2)==9
            assert p['REGMODE_'+port]=='NOREG' and p['WRITEMODE_'+port]=='NORMAL'
            assert p['CLK'+port+'MUX']=='CLK'+port and p['CSDECODE_'+port]=='0b000'
            assert c['CLK'+port]==top['ports']['clk']['bits']
            assert c['RST'+port]==['0'] and c['OCE'+port]==['1']
            for bit in range(3):assert c['CS'+port+str(bit)]==['0'] and c['AD'+port+str(bit)]==['0']
        assert all(int(value,2)==0 for name,value in p.items() if name.startswith('INITVAL_'))


def ram_library(library):
    lib=Path(library).read_text(encoding='utf-8')
    start=lib.index('module DP16KD(');end=lib.index('endmodule',start)
    pins=lambda prefix,high,low: ','.join(prefix+str(i) for i in range(high,low-1,-1))
    body='''
        reg [8:0] memory [0:2047];
        reg [8:0] qa=0,qb=0;
        wire [10:0] aa={ADDR_A},ab={ADDR_B};
        wire [8:0] da={DATA_A},db={DATA_B};
        wire ena=CEA && {CSA2,CSA1,CSA0}==0;
        wire enb=CEB && {CSB2,CSB1,CSB0}==0;
        assign {OUT_A}={9'd0,qa};
        assign {OUT_B}={9'd0,qb};
        integer row;
        initial begin
            if(DATA_WIDTH_A!=9 || DATA_WIDTH_B!=9 || REGMODE_A!="NOREG" || REGMODE_B!="NOREG" ||
               WRITEMODE_A!="NORMAL" || WRITEMODE_B!="NORMAL" || CLKAMUX!="CLKA" || CLKBMUX!="CLKB" ||
               CSDECODE_A!="0b000" || CSDECODE_B!="0b000") $fatal(1,"Unsupported DP16KD model profile");
            for(row=0;row<2048;row=row+1) memory[row]=0;
        end
        always @(posedge CLKA) begin
            if(RSTA) qa<=0;
            else if(ena) begin if(WEA) memory[aa]<=da;else qa<=memory[aa];end
            if(ena && enb && aa==ab && (WEA || WEB)) $fatal(1,"Mapped RAM port collision");
        end
        always @(posedge CLKB) begin
            if(RSTB) qb<=0;
            else if(enb) begin if(WEB) memory[ab]<=db;else qb<=memory[ab];end
        end
    '''
    for token,prefix,high,low in [('ADDR_A','ADA',13,3),('ADDR_B','ADB',13,3),('DATA_A','DIA',8,0),
        ('DATA_B','DIB',8,0),('OUT_A','DOA',17,0),('OUT_B','DOB',17,0)]:body=body.replace(token,pins(prefix,high,low))
    return lib[:end]+body+lib[end:]
