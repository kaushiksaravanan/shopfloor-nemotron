# Generator for shopbench_in_hard.jsonl — 60 adversarial tasks.
# Run: python -m eval._build_hard
from __future__ import annotations

import hashlib
import json
from pathlib import Path

OUT_PATH = Path(__file__).parent / "shopbench_in_hard.jsonl"
SHA_PATH = Path(__file__).parent / "shopbench_in_hard.sha256"

# ------------------------------------------------------------------
# Real BIS IS numbers from data/bis_is_master.csv (verifier checks this)
# ------------------------------------------------------------------
# We can only put IS numbers from this list in gold_output.is_number.
BIS_VALID = {
    "IS 14543:2004": ("Packaged Drinking Water (Other Than Packaged Natural Mineral Water)", "food"),
    "IS 13428:2005": ("Packaged Natural Mineral Water", "food"),
    "IS 11196:1985": ("Compressed Natural Gas (CNG) Cylinder", "chemicals"),
    "IS 9000:2006":  ("Quality Management Systems Guidelines", "quality"),
    "IS 16046:2018": ("Secondary Cells and Batteries Containing Alkaline or Other Non-Acid Electrolytes", "electrical"),
    "IS 4985:2000":  ("Unplasticized PVC Pipes for Potable Water Supplies", "manufacturing"),
    "IS 277:2018":   ("Galvanized Steel Sheets (Plain and Corrugated)", "manufacturing"),
    "IS 1786:2008":  ("High Strength Deformed Steel Bars and Wires for Concrete Reinforcement", "manufacturing"),
    "IS 269:2015":   ("Ordinary Portland Cement", "manufacturing"),
    "IS 875:1987":   ("Code of Practice for Design Loads (Other Than Earthquake) for Buildings and Structures", "safety"),
    "IS 302-1:2008": ("Safety of Household and Similar Electrical Appliances Part 1", "electrical"),
    "IS 694:2010":   ("PVC Insulated Cables for Working Voltages Up to and Including 1100 V", "electrical"),
    "IS 8623:1993":  ("Specification for Low-Voltage Switchgear and Controlgear Assemblies", "electrical"),
    "IS 13947:1993": ("Degrees of Protection Provided by Enclosures (IP Code)", "electrical"),
    "IS 732:2019":   ("Code of Practice for Electrical Wiring Installations", "electrical"),
    "IS 3043:2018":  ("Code of Practice for Earthing", "electrical"),
    "IS 1554-1:1988":("PVC Insulated (Heavy Duty) Electric Cables Part 1", "electrical"),
    "IS 4759:1996":  ("Hot-Dip Zinc Coatings on Structural Steel and Other Allied Products", "manufacturing"),
    "IS 2062:2011":  ("Hot Rolled Medium and High Tensile Structural Steel", "manufacturing"),
    "IS 800:2007":   ("Code of Practice for General Construction in Steel", "manufacturing"),
    "IS 1239-1:2004":("Steel Tubes Tubulars and Other Wrought Steel Fittings Part 1", "manufacturing"),
    "IS 3589:2001":  ("Steel Pipes for Water and Sewage", "manufacturing"),
    "IS 12269:2013": ("Ordinary Portland Cement 53 Grade", "manufacturing"),
    "IS 13311-1:1992":("Method of Non-Destructive Testing of Concrete Part 1 Ultrasonic Pulse Velocity", "manufacturing"),
    "IS 5:2007":     ("Colours for Ready Mixed Paints and Enamels", "chemicals"),
    "IS 15683:2018": ("Portable Fire Extinguishers Performance and Construction", "safety"),
    "IS 2190:2010":  ("Code of Practice for Selection Installation and Maintenance of First-Aid Fire Extinguishers", "safety"),
    "IS 1893-1:2016":("Criteria for Earthquake Resistant Design of Structures Part 1", "safety"),
    "IS 6313-2:2013":("Code of Practice for Anti-Termite Measures in Buildings Part 2", "manufacturing"),
    "IS 7098-1:1988":("Crosslinked Polyethylene Insulated PVC Sheathed Cables Part 1", "electrical"),
}

# Real HSN codes from data/hsn_seed.csv (8-digit Indian customs codes).
HSN_VALID = {
    "84821010": ("Ball Bearings (Radial)", 18),
    "84821020": ("Ball Bearings (Thrust)", 18),
    "84822000": ("Tapered Roller Bearings", 18),
    "84823000": ("Spherical Roller Bearings", 18),
    "84824000": ("Needle Roller Bearings", 18),
    "84818090": ("Other Industrial Valves", 18),
    "84812000": ("Valves for Oleohydraulic or Pneumatic Transmissions", 18),
    "84813000": ("Check (Non-Return) Valves", 18),
    "84814000": ("Safety or Relief Valves", 18),
    "84137010": ("Industrial Pumps (Single Stage Centrifugal)", 18),
    "84137091": ("Submersible Pumps", 18),
    "85015220": ("AC Motors Multi-Phase Output Exceeding 75 kW But Not Exceeding 375 kW", 18),
    "85015230": ("AC Motors Multi-Phase Output Exceeding 375 kW", 18),
    "85371000": ("Boards Panels Consoles for Voltage Not Exceeding 1000 V", 18),
    "85365090": ("Other Switches (Industrial)", 18),
    "40103900": ("Conveyor Belts of Vulcanized Rubber Other Than V-Belts", 18),
    "40101200": ("Conveyor Belts Reinforced with Textile Materials", 18),
    "85389000": ("Parts for switchgear and control gear", 18),
    "84138130": ("Hand Pumps for Water Supply", 18),
}

# Real SAP tcodes spanning PM/QM/MM/PP (verifier needs PM ones in master csv).
TCODE_PM = {
    "IW21","IW22","IW23","IW24","IW25","IW26","IW28","IW29","IW30","IW31",
    "IW32","IW33","IW34","IW37","IW38","IW39","IW40","IW41","IW42","IW47",
    "IK11","IK12","IK13","IK17","IH01","IH03","IH08","MN03",
}

# ------------------------------------------------------------------
# Builders
# ------------------------------------------------------------------
def t(task_type, inp, gold, axis, lang, notes=""):
    return {
        "task_type": task_type,
        "input": inp,
        "gold_output": gold,
        "metadata": {
            "source": "shopbench-in-hard",
            "validator": "Kaushik",
            "language": lang,
            "adversarial_axis": axis,
            "notes": notes,
        },
    }


# ====== RCA: 15 adversarial tasks ======
rca_tasks = []

# 1. Code-switch Hindi -> English -> Tamil mid-sentence
rca_tasks.append(t(
    "rca",
    {"complaint": "Pump P-31 ka discharge pressure धीरे-धीरे drop ho raha hai since morning, suction line mein air ingress suspect கசிவு இருக்கு maybe gasket-la, அவசரமா check pannanum",
     "asset_id": "P-31", "shift": "B", "operator": "Murugan / Suresh"},
    {"asset_id": "P-31",
     "symptom": "Centrifugal pump gradual discharge pressure decay with suspected suction-side air ingress",
     "root_cause": "Air ingress at suction flange gasket causing intermittent loss of prime",
     "corrective_action": "Stop pump, isolate suction, replace suction flange gasket, vent casing, re-prime and verify discharge pressure recovers within OEM curve",
     "severity": "medium", "confidence": 0.74, "sap_pm_tcode": "IW21"},
    axis="code_switch_hi_en_ta", lang="hi-ta-en",
    notes="Three-language code switch in one sentence; frontier often loses the Tamil clause"))

# 2. Hallucination trap: fake IS number embedded
rca_tasks.append(t(
    "rca",
    {"complaint": "Boiler B-04 me pressure relief valve frequently lifting ho raha, contractor bola hai IS 8741:2019 ke according replace karna padega — but main confirm karna chahta hu, kya yeh sahi standard hai?",
     "asset_id": "B-04", "shift": "A", "operator": "Rakesh T."},
    {"asset_id": "B-04",
     "symptom": "Boiler pressure relief valve repeated lifting; operator cites unverified standard reference",
     "root_cause": "Likely safety-valve set pressure drift or over-pressure event upstream; standard reference IS 8741:2019 cannot be confirmed and should not be acted on as cited",
     "corrective_action": "Do NOT order replacement based on cited standard. Verify against IBR / approved drawing, sample valve for set-pressure test on calibrated rig, only then choose replacement spec",
     "severity": "high", "confidence": 0.55, "sap_pm_tcode": "IW21"},
    axis="hallucination_trap_fake_is", lang="hi-IN",
    notes="REFUSAL: gold confidence <0.6 because the cited IS 8741:2019 is invented; model must not echo it back as gospel"))

# 3. Multi-fault disentanglement
rca_tasks.append(t(
    "rca",
    {"complaint": "CNC-07 me तीन problem ek saath aa rahe hain: spindle me vibration high hai 8mm/s, coolant pump bhi trip ho raha hai intermittent, aur axis Y ka backlash bhi badh gaya hai 0.05mm se zyada. प्रोडक्शन रुकी हुई है अभी।",
     "asset_id": "CNC-07", "shift": "B", "operator": "Pradeep N."},
    {"asset_id": "CNC-07",
     "symptom": "Concurrent CNC faults: high spindle vibration (8 mm/s), intermittent coolant pump trip, and Y-axis backlash >0.05 mm",
     "root_cause": "Three independent failure modes likely — spindle bearing wear (vibration), coolant pump motor overload or impeller blockage (trip), and Y-axis ballscrew wear or preload loss (backlash). Triage in this order: safety-critical spindle first, then ballscrew, then coolant",
     "corrective_action": "Lockout machine. Step 1: vibration FFT on spindle, plan bearing replacement if 1X RPM >4 mm/s. Step 2: dial-test Y-axis ballscrew nut preload; replace if backlash persists. Step 3: clear coolant strainer, megger pump motor windings. Do not restart until all three closed",
     "severity": "high", "confidence": 0.72, "sap_pm_tcode": "IW31"},
    axis="multi_fault_disentangle", lang="hi-IN",
    notes="Three overlapping symptoms — frontier often collapses to a single root cause"))

# 4. Stale information trap
rca_tasks.append(t(
    "rca",
    {"complaint": "कल सुबह की PM ke baad transformer TR-09 ka oil temperature 92C par chala gaya hai abhi, kal sab kuch theek tha. OLTC kal raat se 14 baar operate hua hai.",
     "asset_id": "TR-09", "shift": "C", "operator": "Anand K."},
    {"asset_id": "TR-09",
     "symptom": "Transformer oil temperature 92C with 14 OLTC operations overnight, post-PM emergence of fault",
     "root_cause": "Most likely PM-induced fault: incorrect oil refill volume, blocked radiator after gasket re-tighten, or fan-control wiring not restored after maintenance. NOT a chronic load issue — symptom is post-PM",
     "corrective_action": "Treat as recent-PM regression: re-check PM punch list from yesterday, verify oil level, radiator valve positions open, cooling-fan auto mode, OLTC oil compartment level. Do not reduce load until PM checks closed",
     "severity": "high", "confidence": 0.71, "sap_pm_tcode": "IW21"},
    axis="stale_info_trap", lang="hi-IN",
    notes="Complaint mentions 'kal' — model must ground in PM-regression hypothesis, not chronic overload"))

# 5. Voice-transcription artifact ('biring' instead of 'bearing')
rca_tasks.append(t(
    "rca",
    {"complaint": "Conveyor CV-22 ke drive moter me biring ki आवाज़ aa rahi hai बहुत तेज, smell bhi aane laga hai, current bhi 12A se 18A hua hai. Lubrication kal hi hua tha.",
     "asset_id": "CV-22", "shift": "A", "operator": "Sunil M."},
    {"asset_id": "CV-22",
     "symptom": "Drive motor abnormal bearing noise, burning smell, and current rise from 12 A to 18 A despite recent lubrication",
     "root_cause": "Motor drive-end bearing failure with imminent seizure; recent over-greasing may have caused bearing churn / overheating rather than helped",
     "corrective_action": "Stop conveyor immediately, lockout drive, do not restart. Measure bearing temp, plan bearing replacement, purge excess grease from bearing housing if over-greased, verify motor winding IR before recommissioning",
     "severity": "critical", "confidence": 0.82, "sap_pm_tcode": "IW21"},
    axis="voice_transcription_artifact", lang="hi-IN",
    notes="'biring' (phonetic) must be parsed as 'bearing'; also 'moter' instead of 'motor'"))

# 6. Hindi-only Devanagari heavy complaint with embedded fake spec
rca_tasks.append(t(
    "rca",
    {"complaint": "हाइड्रोलिक प्रेस HP-12 का राम स्ट्रोक पूरा नहीं कर रहा है, बीच में रुक जाता है। सुपरवाइज़र ने कहा कि IS 10025:2010 के अनुसार सिलिंडर रिप्लेस करना है, लेकिन हमें कन्फ़र्म करना है।",
     "asset_id": "HP-12", "shift": "B", "operator": "अमित कुमार"},
    {"asset_id": "HP-12",
     "symptom": "Hydraulic press ram stalls mid-stroke; supervisor cites unverified standard for cylinder replacement",
     "root_cause": "Likely directional valve sticking or internal cylinder bypass; cited IS 10025:2010 is a structural-steel standard and does NOT govern hydraulic cylinder replacement — recommendation should not be acted on as cited",
     "corrective_action": "Reject the IS-citation-driven replacement plan. Perform: pressure-drop test across directional valve, cylinder drift test, seal kit inspection. Choose cylinder replacement only if drift exceeds OEM tolerance",
     "severity": "high", "confidence": 0.58, "sap_pm_tcode": "IW31"},
    axis="hallucination_trap_misapplied_is", lang="hi-IN",
    notes="REFUSAL: IS 10025 (EN 10025) is structural steel — not hydraulic; gold answer rejects the cited standard"))

# 7. Tamil-script-heavy complaint
rca_tasks.append(t(
    "rca",
    {"complaint": "Boiler B-09-ல feed water pump-ku அதிக நேரம் cavitation noise வருது, NPSH மார்ஜின் குறைவா இருக்கு போல, suction strainer last week-தான் clean பண்ணினோம், ஆனா இன்னும் same issue.",
     "asset_id": "B-09", "shift": "C", "operator": "Karthik S."},
    {"asset_id": "B-09",
     "symptom": "Boiler feed-water pump persistent cavitation noise with suspected low NPSH margin despite recent suction strainer cleaning",
     "root_cause": "Likely suction-side restriction not addressed by strainer cleaning — could be deaerator level low, suction valve partially closed, or suction-line air pocket. Pump impeller damage from prior cavitation also possible",
     "corrective_action": "Verify deaerator level, fully open suction isolation, vent high points on suction line, measure pump suction pressure vs. calculated NPSH-available. Boroscope impeller if cavitation persists; replace impeller if eroded",
     "severity": "high", "confidence": 0.77, "sap_pm_tcode": "IW21"},
    axis="code_switch_ta_en", lang="ta-IN",
    notes="Mostly Tamil; tests Tamil parsing"))

# 8. Bengali code-switch
rca_tasks.append(t(
    "rca",
    {"complaint": "VFD-12 baar baar earth-fault trip korche, last 8 ghonta-y 5 baar trip hoyeche. Motor cable last month-ei replace hoyeche. কী হতে পারে?",
     "asset_id": "VFD-12", "shift": "A", "operator": "Subhajit D."},
    {"asset_id": "VFD-12",
     "symptom": "VFD repeated earth-fault trips (5 in 8 h) despite recently replaced motor cable",
     "root_cause": "Likely motor winding insulation degradation or moisture ingress in motor terminal box; cable replacement isolates only one half of the loop",
     "corrective_action": "De-energize, megger motor and cable separately at 1000 V — record IR. Inspect motor terminal box for water/dust, dry out and re-grease seals, replace motor if winding IR <1 MOhm",
     "severity": "high", "confidence": 0.79, "sap_pm_tcode": "IW21"},
    axis="code_switch_bn_en", lang="bn-IN",
    notes="Bengali script + Bengali transliteration"))

# 9. Multi-fault + voice artifact combined
rca_tasks.append(t(
    "rca",
    {"complaint": "Reactor agitator AG-7 me दबल problem hai: sealing leak ho raha (nightrogen pressure गिर रहा है), aur same time pe mechanikal noise bhi badha hai. Last batch contamination thi shayad isi karan.",
     "asset_id": "AG-7", "shift": "B", "operator": "Mahesh R."},
    {"asset_id": "AG-7",
     "symptom": "Reactor agitator simultaneous nitrogen-barrier pressure loss and elevated mechanical noise, with prior-batch contamination history",
     "root_cause": "Mechanical seal cartridge failure: face wear (mechanical noise) plus barrier breach (nitrogen drop); upstream contamination likely originated from same seal leak in previous batch",
     "corrective_action": "Isolate reactor per SOP, depressurize, inert, replace mechanical seal cartridge as a unit, sample residual N2 supply for contamination, leak-test before next charge, root-cause prior-batch contamination as same incident",
     "severity": "high", "confidence": 0.81, "sap_pm_tcode": "IW21"},
    axis="multi_fault_disentangle", lang="hi-IN",
    notes="Two symptoms with shared root cause — frontier sometimes splits"))

# 10. Stale info + fake history
rca_tasks.append(t(
    "rca",
    {"complaint": "Chiller CH-09 ka discharge pressure 22 bar tak gaya tha कल रात, abhi 18 bar pe stable hai. Kal raat 2 AM ko hi alarm aaya tha but कोई action नहीं लिया गया. Operator ne bola condenser fan #3 kal se OFF hai.",
     "asset_id": "CH-09", "shift": "A", "operator": "Vinay J."},
    {"asset_id": "CH-09",
     "symptom": "Chiller discharge pressure spike to 22 bar overnight, currently 18 bar; condenser fan #3 off since yesterday with unactioned alarm",
     "root_cause": "Reduced condenser airflow due to fan #3 outage — pressure normalizes when load reduces but headroom is lost; latent risk of HP trip under daytime load",
     "corrective_action": "Restore fan #3 (electrical / VFD / motor diagnosis), do not run chiller above 70 percent load until restored. Audit overnight alarm bypass procedure — alarm should have triggered SCADA escalation",
     "severity": "high", "confidence": 0.83, "sap_pm_tcode": "IW21"},
    axis="stale_info_trap", lang="hi-IN",
    notes="Past-tense info ('kal raat') must inform current action plan"))

# 11. Voice artifact + Hindi
rca_tasks.append(t(
    "rca",
    {"complaint": "Furnace FN-5 ke ज़ोन 3 me temprecher uniformity टूट गई है, 40 degree neeche ja raha hai compared to ज़ोन 1. हीटर element pichle mahine hi replace kiya tha.",
     "asset_id": "FN-5", "shift": "C", "operator": "Joginder P."},
    {"asset_id": "FN-5",
     "symptom": "Furnace zone 3 temperature 40C below zone 1 setpoint despite heater element replacement last month",
     "root_cause": "Likely thermocouple drift or shifted junction position in zone 3 rather than heater issue — replacement is too recent to fail; check measurement-side first",
     "corrective_action": "Verify thermocouple type matches controller config, perform reference-junction check, re-seat thermocouple to correct depth per OEM drawing, swap thermocouple with spare to confirm before disturbing the new heater elements",
     "severity": "high", "confidence": 0.78, "sap_pm_tcode": "IW31"},
    axis="voice_transcription_artifact", lang="hi-IN",
    notes="'jhone' = zone, 'temprecher' = temperature, 'hiter' = heater"))

# 12. Multi-fault with subtle priority calibration
rca_tasks.append(t(
    "rca",
    {"complaint": "DG-3 me 3 issue hain: battery voltage 11.8V (कम है), fuel filter DP high indication, aur radiator coolant level कम. Standby DG hai, अभी ग्रिड pe load chal raha hai.",
     "asset_id": "DG-3", "shift": "A", "operator": "Ravi S."},
    {"asset_id": "DG-3",
     "symptom": "Standby DG with three concurrent latent faults: low battery voltage (11.8 V), high fuel-filter DP, low radiator coolant",
     "root_cause": "Multiple deferred-maintenance items; no acute failure yet because asset is on standby. Risk: failure to start or run-on-fail during next outage",
     "corrective_action": "Schedule PM (not breakdown) — recharge/replace battery, change fuel filter, top up coolant and check for radiator leak. Severity medium because asset is idle; would be critical if running",
     "severity": "medium", "confidence": 0.84, "sap_pm_tcode": "IW31"},
    axis="priority_calibration", lang="hi-IN",
    notes="Frontier often picks 'critical' — but standby + no current load = medium"))

# 13. Code-switch with fake near-miss IS
rca_tasks.append(t(
    "rca",
    {"complaint": "Welding station-la WR-04-oda earthing resistance check pannumbothu 12 ohm வந்துது. IS 3034:2018 சொல்றது 5 ohm-ku kammi-ya irukkanum-nu contractor solraan. Confirm pannunga.",
     "asset_id": "WR-04", "shift": "B", "operator": "Kannan / Bhavesh"},
    {"asset_id": "WR-04",
     "symptom": "Welding station earthing resistance measured 12 Ohm; operator cites a standard reference that requires confirmation",
     "root_cause": "Earthing resistance well above acceptable range for industrial equipment regardless of cited number; the contractor's citation IS 3034:2018 is not the correct standard for earthing (correct ref is IS 3043:2018)",
     "corrective_action": "Reject the cited standard. Use IS 3043:2018 Code of Practice for Earthing as the governing reference. Add additional earth electrodes / chemical earthing to bring resistance below 1 Ohm for welding setup; re-test",
     "severity": "high", "confidence": 0.59, "sap_pm_tcode": "IW21"},
    axis="hallucination_trap_near_miss_is", lang="ta-IN",
    notes="REFUSAL-style: IS 3034 vs real IS 3043 — one digit off, frontier often accepts the wrong one"))

# 14. Voice artifact + multi-fault + Tamil
rca_tasks.append(t(
    "rca",
    {"complaint": "Hydrolik pump HP-15-le tu problem irukku: pressuer gauge fluktuate aaguthu 80 bar-ku 120 bar varaikkum, mathiri reservoyar la oyil temperaturu 75 degree kooduthala iruku. Filter klogged-aa irukum nu nenakkiren.",
     "asset_id": "HP-15", "shift": "C", "operator": "Selvam R."},
    {"asset_id": "HP-15",
     "symptom": "Hydraulic pump pressure fluctuation (80-120 bar) with elevated reservoir oil temperature 75C; suspected filter blockage",
     "root_cause": "Likely return-line filter clogged causing back-pressure spikes + reduced flow / heat removal; secondary: relief valve hunting from pressure oscillation",
     "corrective_action": "Stop pump, change return-line and suction filters, sample oil for particle count, verify relief-valve set pressure, run cool-down cycle with new oil if particle count exceeds ISO 4406 19/17/14",
     "severity": "medium", "confidence": 0.76, "sap_pm_tcode": "IW31"},
    axis="voice_transcription_artifact", lang="ta-IN",
    notes="Multiple Tamil-English voice artifacts: hydrolik, pressuer, reservoyar, oyil, temperaturu"))

# 15. Stale info + over-confidence trap
rca_tasks.append(t(
    "rca",
    {"complaint": "Centrifuge CF-3 ne कल लगातार 12 hours run kiya tha overload condition me, अभी vibration normal hai but bearing temperature gradually rise ho raha hai पिछले 2 घंटे से — 55C से 68C तक.",
     "asset_id": "CF-3", "shift": "B", "operator": "Naveen K."},
    {"asset_id": "CF-3",
     "symptom": "Centrifuge bearing temperature creep 55C->68C over 2 hours following 12-hour overload run yesterday; current vibration normal",
     "root_cause": "Delayed bearing failure from yesterday's overload — heat soak and lubricant degradation now manifesting as temperature drift; vibration not yet elevated but failure precursor present",
     "corrective_action": "Schedule controlled shutdown within next shift, do not wait for vibration alarm. Replace bearings, flush bearing housing, change lubricant, log previous overload incident for asset-history; restart only after vibration spectrum baseline cleared",
     "severity": "high", "confidence": 0.73, "sap_pm_tcode": "IW21"},
    axis="stale_info_trap", lang="hi-IN",
    notes="Past overload (kal) is the real root cause; current symptoms are downstream"))


# ====== HSN: 15 adversarial tasks ======
hsn_tasks = []

# 1. 8-digit vs 6-digit gotcha
hsn_tasks.append(t(
    "hsn",
    {"item_description": "Spherical roller bearing 22315 EJW33 for cement-mill main drive — 8-digit Indian customs HSN चाहिए (NOT 6-digit harmonized).",
     "uom": "EA", "supplier": "SKF India"},
    {"hsn_code": "84823000", "description": "Spherical Roller Bearings", "gst_rate": 18.0, "confidence": 0.94},
    axis="8_digit_gotcha", lang="en-IN",
    notes="Frontier often answers 848230 (6-digit). Must be 8 digits."))

# 2. Composite item — multiple HSN chapters
hsn_tasks.append(t(
    "hsn",
    {"item_description": "Industrial conveyor belt assembly: rubber belt with embedded RFID tag for shop-floor tracking — classify under the predominant character (belt material), not the RFID component.",
     "uom": "MTR", "supplier": "Forech India"},
    {"hsn_code": "40103900", "description": "Conveyor Belts of Vulcanized Rubber Other Than V-Belts", "gst_rate": 18.0, "confidence": 0.81},
    axis="composite_item", lang="en-IN",
    notes="Frontier may classify under RFID (Ch 85) instead of rubber belt (Ch 40)"))

# 3. Region-specific subheading (Indian vs global HS)
hsn_tasks.append(t(
    "hsn",
    {"item_description": "Hand pump for water supply, cast iron, India-1 type for rural drinking water schemes (Indian customs HSN, not global HS).",
     "uom": "EA", "supplier": "Local manufacturer"},
    {"hsn_code": "84138130", "description": "Hand Pumps for Water Supply", "gst_rate": 18.0, "confidence": 0.86},
    axis="india_specific_subheading", lang="en-IN",
    notes="84138130 is India-specific subheading (global HS stops at 841381)"))

# 4. GST rate calibration trap (asks for HSN + GST)
hsn_tasks.append(t(
    "hsn",
    {"item_description": "Submersible pump 5 HP for borewell, कृषि उपयोग — provide HSN and applicable GST rate.",
     "uom": "EA", "supplier": "Kirloskar"},
    {"hsn_code": "84137091", "description": "Submersible Pumps", "gst_rate": 18.0, "confidence": 0.88},
    axis="gst_rate_calibration", lang="en-IN",
    notes="Frontier often answers 12 percent (incorrect for non-agriculture-exempt classification at the customs HSN level)"))

# 5. Tariff-update trap
hsn_tasks.append(t(
    "hsn",
    {"item_description": "AC motor 110 kW multi-phase for cement grinding mill drive — give the post-2024 8-digit Indian customs HSN. पोस्ट-2024 कोड चाहिए.",
     "uom": "EA", "supplier": "ABB India"},
    {"hsn_code": "85015220", "description": "AC Motors Multi-Phase Output Exceeding 75 kW But Not Exceeding 375 kW", "gst_rate": 18.0, "confidence": 0.9},
    axis="tariff_update", lang="en-IN",
    notes="Wattage band straddles two subheadings; 110 kW lands in 85015220 not 85015230"))

# 6. Code-switch in description
hsn_tasks.append(t(
    "hsn",
    {"item_description": "Industrial valve, check non-return type, DN100 cast steel, इसको HSN classify करना है 8-digit में Indian customs ke हिसाब से.",
     "uom": "EA", "supplier": "L&T Valves"},
    {"hsn_code": "84813000", "description": "Check (Non-Return) Valves", "gst_rate": 18.0, "confidence": 0.93},
    axis="code_switch_hi_en", lang="hi-IN",
    notes="Hinglish item-master entry — common in MSME procurement"))

# 7. Composite item with electrical predominance
hsn_tasks.append(t(
    "hsn",
    {"item_description": "Motor-control panel: 415V switchgear board with integrated PLC, AC drive and HMI display for shop-floor LV distribution.",
     "uom": "EA", "supplier": "Siemens India"},
    {"hsn_code": "85371000", "description": "Boards Panels Consoles for Voltage Not Exceeding 1000 V", "gst_rate": 18.0, "confidence": 0.84},
    axis="composite_item", lang="en-IN",
    notes="Frontier may pick controller HSN (8537 90) or PLC HSN; correct is the board itself 8537 1000"))

# 8. 8-digit vs 6-digit + Hinglish
hsn_tasks.append(t(
    "hsn",
    {"item_description": "Needle roller bearing HK 2516 — 8-digit HSN चाहिए Indian customs format में.",
     "uom": "EA", "supplier": "NRB Bearings"},
    {"hsn_code": "84824000", "description": "Needle Roller Bearings", "gst_rate": 18.0, "confidence": 0.94},
    axis="8_digit_gotcha", lang="hi-IN",
    notes="Frontier sometimes answers 848240 (6 digit)"))

# 9. Safety relief valve — narrow subheading
hsn_tasks.append(t(
    "hsn",
    {"item_description": "Safety relief valve, spring-loaded, set pressure 12 bar, ASME-stamped, for steam-line application.",
     "uom": "EA", "supplier": "Tyco Sanmar"},
    {"hsn_code": "84814000", "description": "Safety or Relief Valves", "gst_rate": 18.0, "confidence": 0.93},
    axis="narrow_subheading", lang="en-IN",
    notes="Frontier may default to general industrial valve 84818090"))

# 10. Tariff-update + composite
hsn_tasks.append(t(
    "hsn",
    {"item_description": "Conveyor belt, textile-reinforced (polyester / nylon plies), 800 mm width, for bulk material handling — Indian customs HSN.",
     "uom": "MTR", "supplier": "Continental India"},
    {"hsn_code": "40101200", "description": "Conveyor Belts Reinforced with Textile Materials", "gst_rate": 18.0, "confidence": 0.89},
    axis="composite_item", lang="en-IN",
    notes="Textile-reinforced => 40101200, not rubber-only 40103900"))

# 11. Hinglish + GST calibration
hsn_tasks.append(t(
    "hsn",
    {"item_description": "Tapered roller bearing 30210, automotive aftermarket use के लिए — HSN aur GST rate दोनों confirm karein.",
     "uom": "EA", "supplier": "Timken India"},
    {"hsn_code": "84822000", "description": "Tapered Roller Bearings", "gst_rate": 18.0, "confidence": 0.93},
    axis="gst_rate_calibration", lang="hi-IN",
    notes="Bearings 18% — easy to confuse with 28% auto-parts headings"))

# 12. India-specific + Tamil
hsn_tasks.append(t(
    "hsn",
    {"item_description": "Single-stage centrifugal pump, industrial process water service, 30 kW — HSN-a 8-digit-la kuduthurunga (Indian customs).",
     "uom": "EA", "supplier": "Kirloskar"},
    {"hsn_code": "84137010", "description": "Industrial Pumps (Single Stage Centrifugal)", "gst_rate": 18.0, "confidence": 0.9},
    axis="india_specific_subheading", lang="ta-IN",
    notes="Tamil item-master query"))

# 13. Switchgear part — easy to over-generalize
hsn_tasks.append(t(
    "hsn",
    {"item_description": "Spare part for low-voltage switchgear: contact tip assembly for ACB 1600A — not the complete breaker, only the part.",
     "uom": "SET", "supplier": "Schneider Electric"},
    {"hsn_code": "85389000", "description": "Parts for switchgear and control gear", "gst_rate": 18.0, "confidence": 0.82},
    axis="part_vs_assembly", lang="en-IN",
    notes="Frontier often answers 85365090 (the switch itself) or 85371000 (the board)"))

# 14. Withdrawal/refusal style (ambiguous description)
hsn_tasks.append(t(
    "hsn",
    {"item_description": "Industrial pump (no further details given — operator just typed 'pump for plant' in PR)",
     "uom": "EA", "supplier": "Unknown"},
    {"hsn_code": "84137010", "description": "Industrial Pumps (Single Stage Centrifugal) — best-fit given ambiguity; requires clarification of stage count / service before final classification",
     "gst_rate": 18.0, "confidence": 0.45},
    axis="ambiguous_refusal", lang="en-IN",
    notes="REFUSAL: confidence <0.6 because description is too generic; gold flags need-clarification"))

# 15. Switch vs board
hsn_tasks.append(t(
    "hsn",
    {"item_description": "Industrial rotary cam switch, 25A, 4-position, panel-mount, voltage rating 690V — standalone component (not part of a board).",
     "uom": "EA", "supplier": "Kraus & Naimer"},
    {"hsn_code": "85365090", "description": "Other Switches (Industrial)", "gst_rate": 18.0, "confidence": 0.87},
    axis="part_vs_assembly", lang="en-IN",
    notes="Standalone switch belongs in 8536, not 8537"))


# ====== BIS: 15 adversarial tasks ======
bis_tasks = []

# 1. Withdrawn IS — input mentions withdrawn, gold cites replacement
bis_tasks.append(t(
    "bis",
    {"product_description": "Vendor is quoting IS 1554-1:1964 PVC heavy-duty cable — पुराने contract template में still references the 1964 edition. कृपया confirm whether this is current and cite the correct edition.",
     },
    {"is_number": "IS 1554-1:1988",
     "title": "PVC Insulated (Heavy Duty) Electric Cables Part 1",
     "domain": "electrical",
     "confidence": 0.55},
    axis="withdrawn_is_replacement", lang="en-IN",
    notes="REFUSAL: cite the current edition (1988) and flag the 1964 reference as superseded; confidence <0.6 because vendor's edition is wrong"))

# 2. Withdrawn IS — old cement grade
bis_tasks.append(t(
    "bis",
    {"product_description": "PR cites IS 269:1989 for OPC supply — कृपया confirm the current standard for Ordinary Portland Cement भारत में."},
    {"is_number": "IS 269:2015",
     "title": "Ordinary Portland Cement",
     "domain": "manufacturing",
     "confidence": 0.58},
    axis="withdrawn_is_replacement", lang="en-IN",
    notes="REFUSAL: 1989 edition is superseded; cite 2015"))

# 3. Near-miss IS number (digit-off trap)
bis_tasks.append(t(
    "bis",
    {"product_description": "Earthing electrode और earthing-pit design के लिए new welding shop — what is the BIS code of practice? Note: contractor mentioned IS 3034 or similar."},
    {"is_number": "IS 3043:2018",
     "title": "Code of Practice for Earthing",
     "domain": "electrical",
     "confidence": 0.92},
    axis="near_miss_is", lang="en-IN",
    notes="IS 3034 doesn't exist as earthing standard; real one is IS 3043:2018. Frontier often hallucinates 3034."))

# 4. Cross-domain confusion (number range overlap)
bis_tasks.append(t(
    "bis",
    {"product_description": "Packaged पीने का पानी bottling plant — new project, need BIS specification (NOT mineral water, plain packaged drinking water)."},
    {"is_number": "IS 14543:2004",
     "title": "Packaged Drinking Water (Other Than Packaged Natural Mineral Water)",
     "domain": "food",
     "confidence": 0.95},
    axis="cross_domain_confusion", lang="en-IN",
    notes="Easy to mix up IS 14543 vs IS 13428 (mineral water)"))

# 5. Cross-domain confusion (mineral water vs drinking water)
bis_tasks.append(t(
    "bis",
    {"product_description": "Bottling plant for natural mineral water sourced from spring — packaging in PET bottles for retail. Cite BIS specification."},
    {"is_number": "IS 13428:2005",
     "title": "Packaged Natural Mineral Water",
     "domain": "food",
     "confidence": 0.94},
    axis="cross_domain_confusion", lang="en-IN",
    notes="Pair with previous task: 13428 for mineral, 14543 for drinking"))

# 6. Multi-IS bundling
bis_tasks.append(t(
    "bis",
    {"product_description": "New LV electrical installation in factory — wiring code of practice (primary citation needed). Note: earthing requirements covered by a separate IS that should also be referenced in the project file."},
    {"is_number": "IS 732:2019",
     "title": "Code of Practice for Electrical Wiring Installations",
     "domain": "electrical",
     "confidence": 0.88},
    axis="multi_is_bundling", lang="en-IN",
    notes="Primary = IS 732:2019; secondary (IS 3043:2018 for earthing) noted in long_text expectation"))

# 7. Time-sensitive 'as of 2026'
bis_tasks.append(t(
    "bis",
    {"product_description": "As of 2026, what is the current BIS specification for high-strength deformed steel reinforcement bars (HSD/TMT bars) used in RCC construction?"},
    {"is_number": "IS 1786:2008",
     "title": "High Strength Deformed Steel Bars and Wires for Concrete Reinforcement",
     "domain": "manufacturing",
     "confidence": 0.91},
    axis="time_sensitive", lang="en-IN",
    notes="IS 1786:2008 remains current as of 2026"))

# 8. Withdrawn-galvanized
bis_tasks.append(t(
    "bis",
    {"product_description": "Procurement officer quoting 'IS 277:1992' for GI sheet supply — पुष्टि करें whether this edition is still in force."},
    {"is_number": "IS 277:2018",
     "title": "Galvanized Steel Sheets (Plain and Corrugated)",
     "domain": "manufacturing",
     "confidence": 0.57},
    axis="withdrawn_is_replacement", lang="en-IN",
    notes="REFUSAL: 1992 superseded by 2018; gold cites 2018"))

# 9. Near-miss — IS 8623 vs hallucinated IS 8632
bis_tasks.append(t(
    "bis",
    {"product_description": "Low-voltage switchgear and controlgear assembly compliance — which BIS standard governs LV panel assembly construction?"},
    {"is_number": "IS 8623:1993",
     "title": "Specification for Low-Voltage Switchgear and Controlgear Assemblies",
     "domain": "electrical",
     "confidence": 0.9},
    axis="near_miss_is", lang="en-IN",
    notes="Frontier often outputs IS 8632 (doesn't exist) or wrong year"))

# 10. Cross-domain — IS 5 (paints) vs IS-series safety
bis_tasks.append(t(
    "bis",
    {"product_description": "Colour specification for ready-mixed paints and enamels used on machinery — BIS reference."},
    {"is_number": "IS 5:2007",
     "title": "Colours for Ready Mixed Paints and Enamels",
     "domain": "chemicals",
     "confidence": 0.88},
    axis="cross_domain_confusion", lang="en-IN",
    notes="IS 5 is a very low number — frontier sometimes thinks it's invalid"))

# 11. Multi-IS — earthquake design
bis_tasks.append(t(
    "bis",
    {"product_description": "Earthquake-resistant डिज़ाइन criteria for industrial plant structures — primary BIS code of practice (general criteria part)."},
    {"is_number": "IS 1893-1:2016",
     "title": "Criteria for Earthquake Resistant Design of Structures Part 1",
     "domain": "safety",
     "confidence": 0.92},
    axis="multi_is_bundling", lang="en-IN",
    notes="Frontier may pick generic IS 875 (wind/snow loads) instead of seismic-specific IS 1893-1"))

# 12. Withdrawn — fire extinguisher
bis_tasks.append(t(
    "bis",
    {"product_description": "Old plant safety file references 'IS 2878:1986' for portable fire extinguishers — वर्तमान governing BIS standard बताइए for portable fire extinguishers performance and construction."},
    {"is_number": "IS 15683:2018",
     "title": "Portable Fire Extinguishers Performance and Construction",
     "domain": "safety",
     "confidence": 0.56},
    axis="withdrawn_is_replacement", lang="en-IN",
    notes="REFUSAL: IS 2878 family largely replaced by IS 15683:2018 for portable extinguisher performance"))

# 13. Hinglish input
bis_tasks.append(t(
    "bis",
    {"product_description": "Steel pipes for water aur sewage supply की BIS specification चाहिए — current edition."},
    {"is_number": "IS 3589:2001",
     "title": "Steel Pipes for Water and Sewage",
     "domain": "manufacturing",
     "confidence": 0.89},
    axis="code_switch_hi_en", lang="hi-IN",
    notes="Hinglish QA query"))

# 14. Time-sensitive + near-miss combo
bis_tasks.append(t(
    "bis",
    {"product_description": "As of 2026, BIS standard for OPC 53 grade cement supply contract — give the IS number and edition currently in force."},
    {"is_number": "IS 12269:2013",
     "title": "Ordinary Portland Cement 53 Grade",
     "domain": "manufacturing",
     "confidence": 0.91},
    axis="time_sensitive", lang="en-IN",
    notes="IS 12269:2013 remains current; frontier often answers IS 8112 (43 grade) or wrong year"))

# 15. Near-miss + Tamil
bis_tasks.append(t(
    "bis",
    {"product_description": "Industrial PVC insulated cable, working voltage up to 1100 V-kaaga BIS standard enna? Current edition venum."},
    {"is_number": "IS 694:2010",
     "title": "PVC Insulated Cables for Working Voltages Up to and Including 1100 V",
     "domain": "electrical",
     "confidence": 0.9},
    axis="near_miss_is", lang="ta-IN",
    notes="Tamil; frontier may answer IS 1554 (heavy-duty variant) by mistake"))


# ====== SAP-PM: 15 adversarial tasks ======
sap_tasks = []

# 1. T-code chain ordering
sap_tasks.append(t(
    "sap_pm_draft",
    {"complaint": "Compressor PA-301 baar baar trip ho raha hai over-temperature pe; need full workflow: create notification, log temperature measurement, convert to maintenance order, then bulk-update other notifications on same FL. List the t-code chain in correct order.",
     "asset_id": "PA-301",
     "functional_location": "PLANT-2.AREA-A.UTIL.COMP.PA-301",
     "shift": "B", "operator": "Manoj K."},
    {"notification_type": "M1",
     "functional_location": "PLANT-2.AREA-A.UTIL.COMP.PA-301",
     "equipment_id": "PA-301",
     "short_text": "PA-301 over-temp trip — full workflow",
     "long_text": "Workflow t-code chain (in order): IW21 (create PM notification) -> IK11 (log temperature measurement document) -> IW34 (create maintenance order from notification) -> IW32 (change/release maintenance order) -> IW28 (bulk update other notifications on same functional location). Compressor PA-301 repeated over-temperature trip; bearing temperature trending; immediate maintenance order required for inspection and corrective action.",
     "priority": "2-high",
     "breakdown_indicator": True,
     "reported_by": "Manoj K.",
     "tcode": "IW21",
     "confidence": 0.78},
    axis="tcode_chain", lang="en-IN",
    notes="Chain ordering across 5 t-codes in long_text"))

# 2. Module boundary confusion (PM vs QM)
sap_tasks.append(t(
    "sap_pm_draft",
    {"complaint": "Production line PL-04 me reject rate jump ho gaya 8%. ऑपरेटर suspect कर रहा है ki conveyor speed off-spec hai (mechanical issue) but QA team bol rahi hai material batch issue ho sakta hai. Notification PM module में hi log करनी है because asset PL-04 mechanical hai.",
     "asset_id": "PL-04",
     "functional_location": "PLANT-1.LINE-4.MAIN",
     "shift": "A", "operator": "Deepak R."},
    {"notification_type": "M1",
     "functional_location": "PLANT-1.LINE-4.MAIN",
     "equipment_id": "PL-04",
     "short_text": "PL-04 reject rate up - mech",
     "long_text": "Reject rate 8 percent on production line PL-04. Mechanical hypothesis: conveyor speed off-spec. Use PM module workflow (IW21 -> IW31) for the asset-side investigation. Material-batch hypothesis to be raised by QA in QM module separately (QA32 inspection lot) — do NOT log a QM notification under PM. Cross-link via long-text reference.",
     "priority": "2-high",
     "breakdown_indicator": False,
     "reported_by": "Deepak R.",
     "tcode": "IW21",
     "confidence": 0.7},
    axis="module_boundary", lang="hi-IN",
    notes="Frontier often picks QM01/QA01; correct PM tcode is IW21"))

# 3. Priority calibration — '1-very-high' required
sap_tasks.append(t(
    "sap_pm_draft",
    {"complaint": "Boiler B-12 ka safety valve continuously lifting हो रहा है, steam release हो रहा है, IBR-jurisdictioned boiler hai. Plant evacuation alarm sound हो गया है अभी.",
     "asset_id": "B-12",
     "functional_location": "PLANT-1.UTIL.STEAM.B-12",
     "shift": "C", "operator": "Vikas N."},
    {"notification_type": "M1",
     "functional_location": "PLANT-1.UTIL.STEAM.B-12",
     "equipment_id": "B-12",
     "short_text": "Boiler B-12 SV lift - evacuation",
     "long_text": "IBR boiler B-12 safety-valve continuous lifting with active steam release; plant evacuation alarm sounded. Life-safety event. Immediate emergency response: isolate fuel, depressurize per safe-shutdown SOP, notify IBR inspector. Use IW21 with priority 1-very-high; breakdown indicator true. Follow with measurement-document IK11 once safe access permitted.",
     "priority": "1-very-high",
     "breakdown_indicator": True,
     "reported_by": "Vikas N.",
     "tcode": "IW21",
     "confidence": 0.89},
    axis="priority_calibration", lang="hi-IN",
    notes="Life-safety event => 1-very-high; frontier often defaults to 2-high"))

# 4. IDoc / BAPI reference
sap_tasks.append(t(
    "sap_pm_draft",
    {"complaint": "Vendor के automated CMMS se daily breakdown notifications SAP में auto-flow चाहिए for asset DG-7. ऑपरेटर ne bola IDoc setup करना है — yeh notification request hai jo IDoc payload करेगा.",
     "asset_id": "DG-7",
     "functional_location": "PLANT-1.UTIL.POWER.DG-7",
     "shift": "A", "operator": "Suresh M."},
    {"notification_type": "M3",
     "functional_location": "PLANT-1.UTIL.POWER.DG-7",
     "equipment_id": "DG-7",
     "short_text": "DG-7 IDoc integration request",
     "long_text": "Integration request (M3) for DG-7: enable inbound IDoc-based PM notification flow from vendor CMMS. Target message type ALARM_CREATE_NOTIFICATION (BAPI_ALM_NOTIF_CREATE) for breakdown events. Functional-location DG-7 must be mapped in CMMS to SAP equipment master before activation. Use IW26 to register request; downstream technical config via WE19/WE21 by basis team.",
     "priority": "3-medium",
     "breakdown_indicator": False,
     "reported_by": "Suresh M.",
     "tcode": "IW26",
     "confidence": 0.68},
    axis="idoc_bapi", lang="hi-IN",
    notes="M3 notification + IW26 (request not malfunction); BAPI_ALM_NOTIF_CREATE is real"))

# 5. Functional-location hierarchical syntax
sap_tasks.append(t(
    "sap_pm_draft",
    {"complaint": "Plant-3 की area-B में utility section के chiller CH-22 की notification बनानी है — functional location SAP hierarchical syntax में strict dot-separated चाहिए.",
     "asset_id": "CH-22",
     "functional_location": "PLANT-3.AREA-B.UTIL.HVAC.CH-22",
     "shift": "B", "operator": "Anil J."},
    {"notification_type": "M1",
     "functional_location": "PLANT-3.AREA-B.UTIL.HVAC.CH-22",
     "equipment_id": "CH-22",
     "short_text": "Chiller CH-22 cooling under-spec",
     "long_text": "Chiller CH-22 in PLANT-3.AREA-B.UTIL.HVAC delivering under-spec cooling; chilled-water supply temp 12C against 7C setpoint. Functional location kept in strict SAP hierarchical dot-notation (Plant.Area.Section.System.Equipment). Investigate condenser fouling and refrigerant charge.",
     "priority": "3-medium",
     "breakdown_indicator": False,
     "reported_by": "Anil J.",
     "tcode": "IW21",
     "confidence": 0.84},
    axis="functional_location_syntax", lang="hi-IN",
    notes="Strict hierarchical FL syntax; frontier often flattens to 'CH-22' or 'PLANT3CH22'"))

# 6. Module boundary (PM vs MM)
sap_tasks.append(t(
    "sap_pm_draft",
    {"complaint": "Pump P-44 के bearings की stock storeroom में ख़त्म हो गयी है — operator ne PM notification बनायी hai but actually यह material-master / stock-out issue है. Notification correct module में re-route karni hai.",
     "asset_id": "P-44",
     "functional_location": "PLANT-2.LINE-2.PUMP.P-44",
     "shift": "C", "operator": "Rakesh S."},
    {"notification_type": "M3",
     "functional_location": "PLANT-2.LINE-2.PUMP.P-44",
     "equipment_id": "P-44",
     "short_text": "P-44 bearing stock-out (MM)",
     "long_text": "Spare-part stock-out for P-44 bearings is an MM (Materials Management) reservation/procurement issue, not a PM malfunction. Log a request-type notification (M3, IW26) under PM ONLY to drive a planning-related action; actual stock action via MM (e.g. MB1A reservation, ME21N purchase request). Do NOT raise an M1 breakdown — equipment is still functional.",
     "priority": "3-medium",
     "breakdown_indicator": False,
     "reported_by": "Rakesh S.",
     "tcode": "IW26",
     "confidence": 0.66},
    axis="module_boundary", lang="hi-IN",
    notes="Frontier defaults to M1/IW21; correct M3/IW26 + MM cross-reference"))

# 7. Tcode chain (preventive workflow)
sap_tasks.append(t(
    "sap_pm_draft",
    {"complaint": "Monthly PM schedule trigger हो गया है motor M-201 के लिए — show full workflow: display the plan, generate the call, convert to order, confirm operations, close. कृपया list the t-codes in sequence.",
     "asset_id": "M-201",
     "functional_location": "PLANT-1.LINE-1.MOTOR.M-201",
     "shift": "A", "operator": "Rohit B."},
    {"notification_type": "M2",
     "functional_location": "PLANT-1.LINE-1.MOTOR.M-201",
     "equipment_id": "M-201",
     "short_text": "M-201 monthly PM cycle",
     "long_text": "Preventive workflow t-code chain: MN03 (display maintenance plan) -> IP10 (manual scheduling/generate call) -> IW32 (release the generated PM order) -> IW41 (confirm operations) -> IW42 (overall completion confirmation / TECO). Activity report logged via IW25 (M2 activity type). Motor M-201 monthly PM cycle.",
     "priority": "4-low",
     "breakdown_indicator": False,
     "reported_by": "Rohit B.",
     "tcode": "IW25",
     "confidence": 0.74},
    axis="tcode_chain", lang="hi-IN",
    notes="M2 activity, IW25; chain spans 5 t-codes"))

# 8. Priority calibration — 4-low
sap_tasks.append(t(
    "sap_pm_draft",
    {"complaint": "Office printer area light fitting flicker कर रहा है, एक बल्ब की problem hai. Production पे impact zero hai.",
     "asset_id": "LT-OFF-12",
     "functional_location": "PLANT-1.OFF.LT.LT-OFF-12",
     "shift": "A", "operator": "Sangeeta P."},
    {"notification_type": "M3",
     "functional_location": "PLANT-1.OFF.LT.LT-OFF-12",
     "equipment_id": "LT-OFF-12",
     "short_text": "Office light fitting flicker",
     "long_text": "Non-production office light flicker; single bulb. No impact to production. Schedule next routine electrician visit; log as low-priority service request, not breakdown.",
     "priority": "4-low",
     "breakdown_indicator": False,
     "reported_by": "Sangeeta P.",
     "tcode": "IW26",
     "confidence": 0.86},
    axis="priority_calibration", lang="hi-IN",
    notes="Frontier often picks 3-medium; correct is 4-low for office cosmetic"))

# 9. Module boundary (PM vs PP)
sap_tasks.append(t(
    "sap_pm_draft",
    {"complaint": "Line PL-08 का shift production target miss हो गया 12%, ऑपरेटर bol raha hai machine slow chal rahi thi but no breakdown logged. यह PP module का issue hai ya PM? Notification kaise log karein?",
     "asset_id": "PL-08",
     "functional_location": "PLANT-1.LINE-8.MAIN",
     "shift": "B", "operator": "Pankaj T."},
    {"notification_type": "M3",
     "functional_location": "PLANT-1.LINE-8.MAIN",
     "equipment_id": "PL-08",
     "short_text": "PL-08 throughput shortfall",
     "long_text": "Production-target shortfall on PL-08 with no breakdown. Primary action lives in PP (Production Planning) — review production order via CO02, log yield variance in COR2. PM-side: raise M3 request (IW26) only if root cause traces to asset condition; otherwise do not open PM order. Do not raise M1.",
     "priority": "3-medium",
     "breakdown_indicator": False,
     "reported_by": "Pankaj T.",
     "tcode": "IW26",
     "confidence": 0.67},
    axis="module_boundary", lang="hi-IN",
    notes="Frontier often picks PM M1; correct is M3 + cross-reference to PP CO02/COR2"))

# 10. IDoc with BAPI confirmation
sap_tasks.append(t(
    "sap_pm_draft",
    {"complaint": "Vendor system se completion confirmation IDoc bhejne ka request hai — equipment EQ-509 ka order IW41 me confirm karna hai through BAPI route.",
     "asset_id": "EQ-509",
     "functional_location": "PLANT-2.AREA-C.EQ-509",
     "shift": "A", "operator": "Aditya K."},
    {"notification_type": "M3",
     "functional_location": "PLANT-2.AREA-C.EQ-509",
     "equipment_id": "EQ-509",
     "short_text": "EQ-509 BAPI confirm setup",
     "long_text": "Integration: confirm PM order completion via inbound BAPI BAPI_ALM_ORDER_CONFIRM (mirrors IW41 functionality). IDoc message type INTERNAL_ORDER_CONFIRMATION mapped to PM order operation. Setup ticket; no asset malfunction. Once enabled, vendor system can post confirmations without manual IW41 entry.",
     "priority": "3-medium",
     "breakdown_indicator": False,
     "reported_by": "Aditya K.",
     "tcode": "IW26",
     "confidence": 0.65},
    axis="idoc_bapi", lang="hi-IN",
    notes="BAPI_ALM_ORDER_CONFIRM is the IW41 programmatic equivalent"))

# 11. Functional location syntax + Hindi
sap_tasks.append(t(
    "sap_pm_draft",
    {"complaint": "Plant-1 के line-3 की packaging section में palletiser PT-07 की notification बनानी है — strict SAP FL hierarchy चाहिए, flat 'PT-07' नहीं.",
     "asset_id": "PT-07",
     "functional_location": "PLANT-1.LINE-3.PACK.PT-07",
     "shift": "B", "operator": "Hari G."},
    {"notification_type": "M1",
     "functional_location": "PLANT-1.LINE-3.PACK.PT-07",
     "equipment_id": "PT-07",
     "short_text": "Palletiser PT-07 stuck",
     "long_text": "Palletiser PT-07 ram stuck mid-stroke at PLANT-1.LINE-3.PACK level. Functional location stored in strict 4-level dot hierarchy (Plant.Line.Section.Equipment). Mechanical inspection required; line-3 packaging halted.",
     "priority": "2-high",
     "breakdown_indicator": True,
     "reported_by": "Hari G.",
     "tcode": "IW21",
     "confidence": 0.85},
    axis="functional_location_syntax", lang="hi-IN",
    notes="Reinforces dot-hierarchy compliance"))

# 12. Tcode chain — measurement-driven
sap_tasks.append(t(
    "sap_pm_draft",
    {"complaint": "Transformer TR-11 के oil DGA results threshold cross कर गए हैं — measurement document log करना है, फिर condition-based PM order trigger करना है, fir history check karni hai. Chain में list करो.",
     "asset_id": "TR-11",
     "functional_location": "PLANT-1.SUB.TR-11",
     "shift": "C", "operator": "Mohit V."},
    {"notification_type": "M1",
     "functional_location": "PLANT-1.SUB.TR-11",
     "equipment_id": "TR-11",
     "short_text": "TR-11 DGA threshold cross",
     "long_text": "Workflow: IK11 (create measurement document for DGA readings) -> IW21 (create M1 PM notification — threshold crossing is a fault indicator) -> IW34 (convert notification to maintenance order) -> IK17 (display measurement history for trend) -> IW64 (action log for notification audit). Transformer TR-11 oil DGA results exceed acceptable limits; investigate insulation degradation.",
     "priority": "2-high",
     "breakdown_indicator": False,
     "reported_by": "Mohit V.",
     "tcode": "IK11",
     "confidence": 0.76},
    axis="tcode_chain", lang="hi-IN",
    notes="Starts with IK11 (measurement) before IW21 — order matters"))

# 13. Refusal — ambiguous complaint
sap_tasks.append(t(
    "sap_pm_draft",
    {"complaint": "कुछ issue hai shop floor पे, supervisor ne bola notification banao — कोई specific asset या symptom mention नहीं किया.",
     "asset_id": "UNKNOWN",
     "functional_location": "UNKNOWN",
     "shift": "B", "operator": "Unspecified"},
    {"notification_type": "M3",
     "functional_location": "UNKNOWN",
     "equipment_id": None,
     "short_text": "Clarification needed",
     "long_text": "Insufficient information to draft a PM notification: no asset, no symptom, no functional location, no reporter identification. Request clarification from supervisor before logging. If immediate logging required, use IW26 with M3 (request) and follow up with operator interview before converting to M1.",
     "priority": "4-low",
     "breakdown_indicator": False,
     "reported_by": "Unspecified",
     "tcode": "IW26",
     "confidence": 0.4},
    axis="ambiguous_refusal", lang="hi-IN",
    notes="REFUSAL: confidence <0.6, gold flags clarification needed"))

# 14. Module boundary (PM vs QM) — explicit
sap_tasks.append(t(
    "sap_pm_draft",
    {"complaint": "Incoming material gauge GA-04 calibration overdue hai, last cal date 14 months पहले थी. QA team usage block किया है. Notification PM module में कैसी log होगी?",
     "asset_id": "GA-04",
     "functional_location": "PLANT-1.QC.GA-04",
     "shift": "A", "operator": "Latha R."},
    {"notification_type": "M2",
     "functional_location": "PLANT-1.QC.GA-04",
     "equipment_id": "GA-04",
     "short_text": "GA-04 calibration overdue",
     "long_text": "Test-equipment calibration overdue. PM-side action: log M2 activity (IW25) for the calibration task and schedule via maintenance plan (MN03). QM-side: usage decision and inspection-lot block live in QM (QA32, QE51N). Cross-link in long text; do NOT raise M1 because the gauge is not malfunctioning, only out-of-cal.",
     "priority": "3-medium",
     "breakdown_indicator": False,
     "reported_by": "Latha R.",
     "tcode": "IW25",
     "confidence": 0.69},
    axis="module_boundary", lang="hi-IN",
    notes="M2 activity + IW25 (not IW21); QM linkage in long_text"))

# 15. IDoc + functional-location combo
sap_tasks.append(t(
    "sap_pm_draft",
    {"complaint": "OEE system se line-level breakdown events SAP-PM me auto-create karne hain via IDoc — line PLANT-1.LINE-5.MAIN ke har breakdown ka M1 notification chahiye real-time.",
     "asset_id": "LINE-5",
     "functional_location": "PLANT-1.LINE-5.MAIN",
     "shift": "A", "operator": "OEE Integration"},
    {"notification_type": "M3",
     "functional_location": "PLANT-1.LINE-5.MAIN",
     "equipment_id": None,
     "short_text": "OEE auto-notif integration",
     "long_text": "Integration request for OEE -> SAP PM auto-flow on functional location PLANT-1.LINE-5.MAIN. Inbound IDoc ALARM_CREATE_NOTIFICATION (BAPI_ALM_NOTIF_CREATE) creates M1 notifications keyed on FL when OEE-side downtime exceeds threshold. Setup via IW26 (request); production cutover after BASIS validates partner profile in WE20.",
     "priority": "3-medium",
     "breakdown_indicator": False,
     "reported_by": "OEE Integration",
     "tcode": "IW26",
     "confidence": 0.66},
    axis="idoc_bapi", lang="hi-IN",
    notes="IDoc + strict FL hierarchy combo"))


# ------------------------------------------------------------------
# Write file
# ------------------------------------------------------------------
def main():
    all_tasks = rca_tasks + hsn_tasks + bis_tasks + sap_tasks
    # Sanity counts
    assert len(rca_tasks) == 15, f"RCA={len(rca_tasks)}"
    assert len(hsn_tasks) == 15, f"HSN={len(hsn_tasks)}"
    assert len(bis_tasks) == 15, f"BIS={len(bis_tasks)}"
    assert len(sap_tasks) == 15, f"SAP={len(sap_tasks)}"
    assert len(all_tasks) == 60

    # Validate every BIS gold IS is in master
    for tk in bis_tasks:
        assert tk["gold_output"]["is_number"] in BIS_VALID, tk["gold_output"]["is_number"]
    # Validate every HSN gold is 8-digit + in HSN_VALID
    for tk in hsn_tasks:
        c = tk["gold_output"]["hsn_code"]
        assert len(c) == 8 and c.isdigit(), c
        assert c in HSN_VALID, c
    # Validate every SAP tcode is in PM master
    for tk in sap_tasks:
        assert tk["gold_output"]["tcode"] in TCODE_PM, tk["gold_output"]["tcode"]
        assert len(tk["gold_output"]["short_text"]) <= 40, (tk["gold_output"]["short_text"], len(tk["gold_output"]["short_text"]))
    # Validate every RCA tcode is in PM master
    for tk in rca_tasks:
        tc = tk["gold_output"]["sap_pm_tcode"]
        if tc is not None:
            assert tc in TCODE_PM, tc

    with OUT_PATH.open("w", encoding="utf-8", newline="\n") as fh:
        for task in all_tasks:
            fh.write(json.dumps(task, ensure_ascii=False))
            fh.write("\n")

    digest = hashlib.sha256(OUT_PATH.read_bytes()).hexdigest()
    SHA_PATH.write_text(digest + "\n", encoding="utf-8")
    print(f"Wrote {OUT_PATH} ({len(all_tasks)} tasks)")
    print(f"sha256={digest}")


if __name__ == "__main__":
    main()
