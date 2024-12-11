use std::collections::HashMap;
use ndarray::{arr1, Array1};


/// # Description
/// Sample data.
pub enum SampleData {
    /// Data for testing `CAPM`
    CAPM,
    /// Data for testing `Portfolio`
    Portfolio,
}

impl SampleData {

    fn load_capm_data(&self) -> (Vec<&str>, HashMap<String, Array1<f64>>) {
        let returns: Array1<f64> = arr1(&[0.0551540121632261, 0.0447766675372913, 0.0970256901094277, 0.0564358892670344, -0.1242128506500266,
            0.1305190299310406, 0.0763942621157174, -0.0164609809886134, 0.072961750485206, 0.1106843772082988, 0.0775539512483334, 0.0987840461332014,
            0.0540097019694354, -0.1147011108835538, -0.0697617158928185, 0.1553735395312054, 0.0850944252266376, 0.1473862162655301, 0.1651315309421344,
            0.2165693951181242, -0.1025264214704002, -0.0600119061155783, 0.0954931260283808, 0.1145736137370141, -0.0055014156624776, -0.0797120540219485,
            0.0073394985214803, 0.0762177475625984, -0.0504969358828828, 0.0991092834112314, 0.0649823898477166, 0.0424891889381942, -0.0680365277256257,
            0.0586573693366816, 0.1050817292007535, 0.0742287096452731, -0.0157123284237963, -0.0540658846712344, 0.0574736773038544, -0.0971308870907154,
            -0.0544960477790734, -0.0814297449519783, 0.1886336825347374, -0.0312080326254287, -0.1209771489521688, 0.1095515711795358, -0.0330275476113557,
            -0.1222724265249621, 0.1105208945377724, 0.0231830575331706, 0.1186485690529139, 0.0289872801620316, 0.0460583194792347, 0.0943301548939321,
            0.0127854047886213, -0.0435052349498845]);
        let market: Array1<f64> = arr1(&[0.0861, 0.0358, 0.0129, 0.0418, -0.0673, 0.0711, 0.0138, -0.0242, 0.0161, 0.0222, 0.0399, 0.0291, 0.0001999999999999,
            -0.0801, -0.1326, 0.1365, 0.0559, 0.0247, 0.0578, 0.0764, -0.0361999999999999, -0.0209, 0.1248, 0.0464, -0.0001999999999999, 0.0278, 0.0308, 0.0493,
            0.0029, 0.0275, 0.0127, 0.0291, -0.0437, 0.0665, -0.0155, 0.0311, -0.0625, -0.0229, 0.0306, -0.0945, -0.0031, -0.0837, 0.0965, -0.0358, -0.0916,
            0.0805999999999999, 0.0489, -0.0608, 0.07, -0.0224, 0.0286999999999999, 0.0096, 0.0070999999999999, 0.0686, 0.0365999999999999, -0.0194]);
        let rf: Array1<f64> = arr1(&[0.0021, 0.0018, 0.0019, 0.0021, 0.0021, 0.0018, 0.0019, 0.0016, 0.0018, 0.0016, 0.0012, 0.0014, 0.0013, 0.0012,
            0.0013, 0.0, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0001, 0.0,
            0.0, 0.0001, 0.0001, 0.0003, 0.0006, 0.0008, 0.0019, 0.0019, 0.0023, 0.0029, 0.0033, 0.0034999999999999, 0.0034, 0.0036, 0.0034999999999999, 0.0036,
            0.004, 0.0045, 0.0045]);
        let smb: Array1<f64> = arr1(&[0.0302999999999999, 0.0174, -0.0351, -0.0116999999999999, -0.0159, 0.0037, -0.0178, -0.0324, 0.0027, 0.0026, 0.0044,
            0.0097, -0.0438, 0.0004, -0.0828, 0.0256, 0.0197, 0.0196, -0.032, -0.0089, 0.0001, 0.0464, 0.0711, 0.0479, 0.0694, 0.0454, -0.0088, -0.0316,
            0.0121999999999999, -0.0034, -0.0461, -0.0068, 0.0112, -0.027, -0.0177, -0.008, -0.0406, 0.0292, -0.0217, -0.0039, -0.0006, 0.013, 0.0182, 0.0152,
            -0.0105, 0.0189, -0.0274, -0.0015, 0.0441, 0.0066, -0.0694, -0.0256, -0.0038, 0.0134, 0.0286, -0.0365]);
        let hml: Array1<f64> = arr1(&[-0.0046, -0.0266999999999999, -0.0417, 0.0215, -0.0237, -0.0070999999999999, 0.0048, -0.0478, 0.0675, -0.0191,
            -0.0202, 0.0175, -0.0625, -0.0381, -0.1387, -0.0133, -0.0487999999999999, -0.022, -0.0141, -0.0297, -0.0271, 0.0425, 0.0209, -0.0151, 0.0301, 0.0715,
            0.0739, -0.0095, 0.0709, -0.0787, -0.0178, -0.0015, 0.0508, -0.0049, -0.0045, 0.0326, 0.1275, 0.0308, -0.0181, 0.0617, 0.0839, -0.0596999999999999,
            -0.0405, 0.0029, 0.0005, 0.0801, 0.0138, 0.0137, -0.0401, -0.0081, -0.0885, -0.0004, -0.0771999999999999, -0.0026, 0.0411, -0.0106]);
        let rmw: Array1<f64> = arr1(&[-0.008, 0.0012, 0.009, 0.0161, -0.0046, 0.009, -0.0008, 0.0056, 0.0184, 0.0044, -0.0159, 0.0001, -0.0116999999999999,
            -0.0147, -0.0157, 0.0272, 0.0095, 0.0009, 0.0040999999999999, 0.0428, -0.0136, -0.0083, -0.022, -0.0196, -0.0378, 0.0039, 0.0636, 0.0243999999999999,
            0.0245, -0.0214, 0.0546, -0.0028, -0.0196, 0.0166, 0.072, 0.0491, 0.0084, -0.0209, -0.0153, 0.0352, 0.0156, 0.0181, 0.0081999999999999, -0.0475,
            -0.0151, 0.0334, 0.0638, 0.0025, -0.0243999999999999, 0.0101, 0.0224, 0.0242, -0.0181, 0.0218, -0.0056999999999999, 0.0343]);
        let cma: Array1<f64> = arr1(&[-0.0151, -0.0159, -0.0095, -0.0222, 0.0177, -0.0044, 0.0036, -0.0068, 0.0337, -0.0096, -0.0124, 0.0124, -0.0232,
            -0.0250999999999999, 0.0124, -0.01, -0.0326, 0.0054, 0.0102, -0.0118999999999999, -0.0185, -0.0078, 0.0139999999999999, -0.0012, 0.0494, -0.0194,
            0.0342, -0.027, 0.0304, -0.0093999999999999, -0.0052, -0.0179, 0.021, -0.0144999999999999, 0.0173, 0.044, 0.0771999999999999, 0.0312, 0.0314,
            0.0588999999999999, 0.0397, -0.0469, -0.0683, 0.0129, -0.008, 0.0664, 0.0318, 0.042, -0.0447, -0.0133, -0.0237, 0.0286, -0.0722, -0.0162,
            0.0056999999999999, -0.0237]);
        let dates: Vec<&str> = vec!["2019-01-31","2019-02-28","2019-03-31","2019-04-30","2019-05-31","2019-06-30","2019-07-31","2019-08-31","2019-09-30",
            "2019-10-31","2019-11-30","2019-12-31","2020-01-31","2020-02-29","2020-03-31","2020-04-30","2020-05-31","2020-06-30","2020-07-31","2020-08-31","2020-09-30",
            "2020-10-31","2020-11-30","2020-12-31","2021-01-31","2021-02-28","2021-03-31","2021-04-30","2021-05-31","2021-06-30","2021-07-31","2021-08-31","2021-09-30",
            "2021-10-31","2021-11-30","2021-12-31","2022-01-31","2022-02-28","2022-03-31","2022-04-30","2022-05-31","2022-06-30","2022-07-31","2022-08-31","2022-09-30",
            "2022-10-31","2022-11-30","2022-12-31","2023-01-31","2023-02-28","2023-03-31","2023-04-30","2023-05-31","2023-06-30","2023-07-31","2023-08-31"];
        let capm_data: HashMap<String, Array1<f64>> = HashMap::from([(String::from("Stock Returns"), returns), (String::from("Market"), market),
                                                                     (String::from("RF"), rf), (String::from("SMB"), smb), (String::from("HML"), hml),
                                                                     (String::from("RMW"), rmw), (String::from("CMA"), cma),]);
        (dates, capm_data)
    }

    fn load_portfolio_data(&self) -> (Vec<&str>, HashMap<String, Array1<f64>>) {
        let bac: Array1<f64> = arr1(&[34.573509216308594, 34.54441452026367, 34.486228942871094, 34.21467590332031, 34.28256607055664, 33.477630615234375,
            33.254573822021484, 33.22547912597656, 33.17699432373047, 33.17699432373047, 33.26427459716797, 33.10910415649414, 32.68938446044922, 33.34336471557617,
            33.27503967285156, 32.21109771728516, 31.781618118286133, 29.809907913208008, 29.54636001586914, 27.82843780517578, 28.072460174560547, 27.808914184570312,
            28.27743911743164, 27.154930114746094, 27.086605072021484, 27.90652275085449, 26.979232788085938, 26.32525062561035, 26.491186141967773, 27.808914184570312,
            27.4477596282959, 27.98461151123047, 27.623455047607425, 27.916284561157227, 27.90652275085449, 27.311107635498047, 26.979232788085938, 27.17445373535156,
            27.272062301635746, 28.023653030395508, 27.799152374267575, 27.87723922729492, 28.81429100036621, 29.643972396850582, 29.82942771911621, 29.341381072998047,
            29.185205459594727, 29.15592384338379, 29.048553466796875, 28.15054702758789, 27.760108947753903, 28.199352264404297, 28.580028533935547,
            28.345766067504883, 27.486804962158203, 27.1939754486084, 26.34477424621582, 27.047557830810547, 27.028039932250977, 26.998756408691406, 
            26.66688346862793, 26.73521041870117, 26.4423828125, 26.98899269104004, 26.705928802490234, 27.88700294494629, 27.789390563964844, 27.437997817993164,
            27.662500381469727, 27.89676284790039, 27.42823982238769, 27.496562957763672, 27.63321685791016, 27.58441162109375, 27.12565040588379, 27.332263946533203,
            28.24727249145508, 28.08001518249512, 28.75889205932617, 29.004863739013672, 28.75889205932617, 28.798250198364254, 28.660505294799805, 28.9261531829834,
            28.65066719055176, 28.896638870239254, 28.719539642333984, 28.404695510864254, 28.10953140258789, 27.509361267089844, 27.30274772644043,
            27.63726806640625, 27.78484916687012, 27.617589950561523, 28.19808006286621, 28.227596282958984, 28.72937774658203, 28.611310958862305,
            27.82420539855957, 28.07017517089844, 28.19808006286621, 28.55227851867676, 28.886798858642575, 29.191802978515625, 28.64082717895508, 28.9261531829834,
            30.205202102661133, 31.02182388305664, 31.179244995117188, 31.464570999145508, 32.12377166748047, 31.63183212280273, 31.88763999938965,
            31.435056686401367, 31.385860443115234, 31.484249114990234, 31.110374450683597, 30.687305450439453, 30.903757095336918, 30.795530319213867,
            31.366182327270508, 30.766014099121094, 30.36262321472168, 30.441333770751957, 30.78569221496582, 30.431493759155277, 29.45745086669922,
            28.81792640686035, 28.808088302612305, 28.64082717895508, 28.68018341064453, 27.98162651062012, 27.9914665222168, 28.15872573852539, 28.040658950805664,
            28.29646873474121, 28.699859619140625, 28.571956634521484, 28.44298553466797, 28.75053024291992, 28.42314338684082, 28.165203094482425,
            27.90726089477539, 28.135440826416016, 28.25448989868164, 28.740610122680664, 28.65132141113281, 28.96879005432129, 28.61163902282715, 28.53227424621582,
            28.42314338684082, 28.32393455505371, 27.82789421081543, 27.421140670776367, 27.381458282470703, 26.95486259460449, 27.0540714263916, 27.341773986816406,
            27.16320037841797, 26.488584518432617, 25.7048397064209, 25.734603881835938, 25.85365104675293, 25.86357307434082, 26.101673126220703, 26.796131134033203,
            26.80605125427246, 26.687000274658203, 26.54810905456543, 26.77628898620605, 27.40130043029785, 27.093753814697266, 26.746524810791016, 26.101673126220703,
            25.36753273010254, 25.26832389831543, 25.34769058227539, 25.91317749023437, 24.97069931030273, 25.486581802368164, 26.13143539428711, 26.19095993041992,
            27.40130043029785, 28.194965362548828, 28.10567855834961, 27.9370231628418, 27.669160842895508, 27.312013626098636, 27.460824966430664, 27.48066711425781,
            28.988630294799805, 29.38546371459961, 29.30609703063965, 29.74261283874512, 29.772375106811523, 29.425146102905277, 29.395383834838867,
            29.494592666625977, 29.325937271118164, 29.29617691040039, 30.06999969482422, 30.489999771118164, 30.959999084472656, 30.81999969482422, 30.57999992370605,
            30.530000686645508, 30.65999984741211, 30.959999084472656, 30.850000381469727, 30.739999771118164, 32.040000915527344, 33.939998626708984,
            33.599998474121094, 33.43000030517578, 33.5099983215332, 32.97999954223633, 33.20000076293945, 33.43000030517578, 33.86000061035156, 33.84000015258789,
            33.880001068115234, 33.66999816894531, 33.900001525878906, 33.529998779296875, 33.79999923706055, 34.43000030517578, 34.15999984741211, 33.630001068115234,
            33.599998474121094, 33.150001525878906, 32.79999923706055, 32.119998931884766, 31.799999237060547, 31.729999542236328, 32.220001220703125,
            32.54999923706055, 32.77000045776367, 32.97999954223633, 33.38999938964844, 33.43000030517578, 33.61000061035156, 34.790000915527344, 34.0099983215332,
            33.54999923706055, 33.470001220703125, 32.9900016784668, 33.040000915527344, 33.18000030517578, 33.119998931884766, 33.06999969482422]);
        let c: Array1<f64> = arr1(&[48.76375198364258, 49.34700775146485, 49.547794342041016, 49.20357894897461, 49.16533279418945, 47.65461349487305,
            47.84584426879883, 48.17094039916992, 47.97970581054688, 48.25699234008789, 48.46734619140625, 49.24182891845703, 49.14621353149414, 50.05455780029297,
            49.89201354980469, 48.84024810791016, 48.45778274536133, 46.4689826965332, 46.22038650512695, 42.77824401855469, 45.32160949707031, 42.854736328125,
            43.6196517944336, 42.30972671508789, 42.15674209594727, 43.09376907348633, 41.79340362548828, 41.544803619384766, 41.2197151184082, 42.81648635864258,
            43.2371940612793, 43.93518829345703, 44.049922943115234, 44.83396530151367, 44.66186141967773, 44.06904983520508, 43.76307678222656, 43.84912872314453,
            44.48019027709961, 45.13037490844727, 44.86264801025391, 45.22599029541016, 47.38689422607422, 47.51118850708008, 47.88409423828125, 48.190059661865234,
            47.23391342163086, 46.88013458251953, 46.88013458251953, 45.7996826171875, 44.80528259277344, 44.96782684326172, 45.49947738647461, 45.6734733581543,
            44.465179443359375, 44.14619064331055, 43.40188217163086, 44.77450180053711, 44.81316757202149, 44.96783065795898, 44.82283401489258, 44.494178771972656,
            43.93353271484375, 44.51351165771485, 43.76920700073242, 45.50915145874024, 44.842166900634766, 44.184852600097656, 44.26218795776367, 44.37818145751953,
            43.00556182861328, 42.75423812866211, 43.11189270019531, 43.20855712890625, 42.84123611450195, 43.3438835144043, 44.77450180053711, 44.69717025756836,
            45.64447784423828, 46.408111572265625, 46.495113372802734, 46.688438415527344, 46.83343505859375, 47.06542205810547, 46.63044357299805, 46.93976593017578,
            46.58210754394531, 46.021461486816406, 45.82813262939453, 45.074161529541016, 44.48451614379883, 44.69717025756836, 44.851837158203125, 44.68750762939453,
            44.74550247192383, 44.50384521484375, 45.18048858642578, 45.23849105834961, 43.86587142944336, 44.21385955810547, 44.14619064331055, 44.97749710083008,
            45.79913711547852, 46.08912658691406, 44.2235221862793, 44.71650314331055, 45.45114517211914, 45.93446350097656, 45.82813262939453, 45.48014831542969,
            46.06978988647461, 45.5188102722168, 45.90546417236328, 46.28245162963867, 45.82813262939453, 46.06978988647461, 45.49947738647461, 44.764835357666016,
            44.66817092895508, 44.69750595092773, 44.81485366821289, 44.1596908569336, 43.45563507080078, 43.357852935791016, 43.58275985717773, 43.1231689453125,
            42.25288391113281, 41.69550704956055, 41.51949691772461, 41.31414794921875, 41.13813400268555, 40.11139297485352, 40.170066833496094, 40.170066833496094,
            40.32652282714844, 40.404747009277344, 40.7958869934082, 40.45363998413086, 40.37541198730469, 40.66876983642578, 40.44386291503906, 40.45363998413086,
            39.82781600952149, 39.69091796875, 39.69091796875, 40.75677490234375, 41.43149185180664, 42.18443298339844, 41.76395797729492, 41.69550704956055,
            41.734622955322266, 41.363040924072266, 40.78610610961914, 40.01360702514648, 40.11139297485352, 39.3291130065918, 39.56379699707031, 40.28740692138672,
            40.21895980834961, 39.60291290283203, 38.977088928222656, 38.92819595336914, 38.99664688110352, 39.671363830566406, 39.83759689331055, 40.52208709716797,
            40.61009216308594, 40.61009216308594, 40.51231002807617, 40.042945861816406, 40.28740692138672, 39.57357406616211, 39.17266082763672, 38.80107498168945,
            38.02857208251953, 38.06768798828125, 37.75477600097656, 38.46860504150391, 37.39297103881836, 38.06768798828125, 38.61528396606445, 38.72284698486328,
            40.43408203125, 41.86046600341797, 41.64254760742188, 41.56330490112305, 41.42462921142578, 41.1076545715332, 41.68217086791992, 42.29631042480469,
            43.940616607666016, 44.4556999206543, 44.2476806640625, 44.93115997314453, 44.83210372924805, 44.66371154785156, 44.59437561035156, 44.802391052246094,
            44.65380859375, 44.43588638305664, 45.317474365234375, 45.664161682128906, 46.78348159790039, 46.922157287597656, 46.25849151611328, 47.40752410888672,
            48.04147338867188, 48.42778778076172, 47.694786071777344, 47.60563659667969, 48.87353515625, 49.7650260925293, 49.35890197753906, 49.19050598144531,
            50.44850158691406, 49.59663009643555, 50.121620178222656, 50.3791618347168, 50.8546257019043, 50.97348785400391, 51.03292465209961, 50.95367813110352,
            52.53855514526367, 53.13288116455078, 53.26165008544922, 53.81636047363281, 53.49938201904297, 52.97439193725586, 52.518741607666016, 51.58763122558594,
            52.12252426147461, 51.37961196899414, 50.87443542480469, 50.626800537109375, 51.03292465209961, 52.3404426574707, 52.79609298706055, 52.7564697265625,
            53.22203063964844, 53.162593841552734, 53.59843826293945, 56.55026245117188, 55.63895797729492, 55.53000259399414, 55.540000915527344, 54.5,
            54.34999847412109, 54.540000915527344, 54.290000915527344, 53.9900016784668]);
        let gs: Array1<f64> = arr1(&[362.5300598144531, 360.35888671875, 362.8402099609375, 358.8274230957031, 357.1796264648437, 350.0263366699219,
            349.80340576171875, 352.3720397949219, 352.6724853515625, 354.3009033203125, 340.8472900390625, 337.9967041015625, 340.7984619140625, 348.5984802246094,
            348.55938720703125, 337.8502502441406, 340.98394775390625, 333.9551696777344, 319.8780822753906, 308.0072631835937, 314.4892883300781, 304.76617431640625,
            307.59722900390625, 296.3218688964844, 302.16943359375, 309.735107421875, 306.2109680175781, 307.3629455566406, 305.1371459960937, 310.9554138183594,
            310.8675537109375, 313.0933532714844, 313.5033874511719, 319.3313293457031, 318.7554016113281, 314.9774475097656, 313.8840637207031, 314.7333984375,
            317.2325134277344, 319.5851745605469, 322.3869323730469, 324.23199462890625, 328.9081115722656, 331.6024780273437, 325.9696655273437, 328.8788452148437,
            330.655517578125, 333.5353698730469, 335.7806701660156, 331.4072265625, 328.556640625, 333.4767761230469, 335.2730712890625, 332.4712829589844,
            325.4425048828125, 320.8347473144531, 313.6205139160156, 319.2434997558594, 318.7456359863281, 317.0470275878906, 314.8797912597656, 313.0933532714844,
            311.90234375, 314.4112243652344, 313.1811828613281, 322.328369140625, 321.7426452636719, 318.3941955566406, 317.1934814453125, 315.8755798339844,
            312.2049865722656, 316.1781921386719, 324.1148376464844, 322.962890625, 318.6053161621094, 311.2279357910156, 318.3594055175781, 316.54949951171875,
            321.4578857421875, 330.2616271972656, 329.9861755371094, 330.5271911621094, 333.94049072265625, 336.9012756347656, 332.8879699707031, 334.1864013671875,
            332.77978515625, 325.3138427734375, 319.7561950683594, 314.3559265136719, 309.5655517578125, 307.2539367675781, 308.3261413574219, 308.53271484375,
            317.8085327148437, 317.2675476074219, 321.27099609375, 314.8182678222656, 307.88348388671875, 310.0180358886719, 311.2967834472656, 315.6346740722656,
            321.064453125, 323.30718994140625, 320.8578796386719, 321.8514099121094, 331.7567443847656, 334.983154296875, 345.1246032714844, 346.2066345214844,
            353.06268310546875, 348.90185546875, 351.715087890625, 348.7149658203125, 347.4559020996094, 350.0527038574219, 351.8724670410156, 345.3606872558594,
            347.38702392578125, 347.446044921875, 349.747802734375, 342.567138671875, 337.08819580078125, 334.8356018066406, 335.16021728515625, 332.2190856933594,
            326.77947998046875, 323.73016357421875, 322.1661376953125, 319.61846923828125, 316.7953796386719, 313.578857421875, 317.0609741210937, 314.729736328125,
            314.9166259765625, 320.6414794921875, 327.1139221191406, 325.9038391113281, 325.0409240722656, 324.73345947265625, 321.1131896972656, 318.5145568847656,
            319.3377685546875, 322.8687744140625, 322.005859375, 328.2049560546875, 332.2517395019531, 341.7636413574219, 339.75018310546875, 340.5238037109375,
            339.10546875, 333.9577331542969, 327.5503234863281, 325.219482421875, 326.1617431640625, 321.1826477050781, 319.327880859375, 322.5513916015625,
            320.9346618652344, 315.90594482421875, 303.62677001953125, 306.08660888671875, 307.9710998535156, 309.9349975585937, 310.0639038085937, 312.2063293457031,
            310.4705810546875, 307.326416015625, 306.7808837890625, 311.8294372558594, 306.84039306640625, 299.50067138671875, 296.75323486328125, 297.606201171875,
            296.7829895019531, 296.7433166503906, 294.343017578125, 294.5711364746094, 287.548828125, 298.3798522949219, 301.13720703125, 304.6583251953125,
            311.2045593261719, 324.95166015625, 321.2719116210937, 321.31158447265625, 321.9165954589844, 318.1673889160156, 322.85888671875, 324.2474670410156,
            335.9612731933594, 334.85040283203125, 333.927978515625, 336.4274597167969, 336.6853332519531, 332.2418212890625, 335.8819274902344, 336.38775634765625,
            334.9595031738281, 334.8999938964844, 340.260009765625, 341.5400085449219, 348.42999267578125, 349.3900146484375, 341.75, 341.9700012207031,
            344.6199951171875, 350.8299865722656, 351.760009765625, 352.6099853515625, 362.7300109863281, 383.4700012207031, 380.510009765625, 376.3999938964844,
            382.4500122070313, 377.5199890136719, 380.5700073242188, 380.6499938964844, 381.6099853515625, 384.4800109863281, 386.4100036621094, 385.7699890136719,
            388.2999877929688, 381.7900085449219, 382.9500122070313, 386.4400024414063, 388.8599853515625, 383.739990234375, 381.9599914550781, 379.75, 377.75,
            380.4500122070313, 377.1799926757813, 376.9100036621094, 382.2000122070313, 385.9599914550781, 380.7699890136719, 379.3999938964844, 382.7000122070313,
            377.7900085449219, 380.55999755859375, 386.8699951171875, 384.010009765625, 383.8500061035156, 387.8599853515625, 383.0199890136719, 384.989990234375,
            386.6600036621094, 385.0400085449219, 384.260009765625]);
        let jpm: Array1<f64> = arr1(&[138.61114501953125, 139.22364807128906, 139.80699157714844, 137.8819580078125, 138.29029846191406, 135.75277709960938,
            134.7124786376953, 135.79165649414062, 137.01666259765625, 138.21253967285156, 139.36947631835938, 138.59169006347656, 137.15280151367188,
            139.6708526611328, 138.8542022705078, 134.7708282470703, 133.9735870361328, 126.7207260131836, 129.9388427734375, 127.60546875, 130.8818817138672,
            124.69847869873048, 127.1193389892578, 122.31652069091795, 123.60958862304688, 126.92491149902344, 123.6484832763672, 123.31792449951172,
            121.44151306152344, 124.92211151123048, 125.30128479003906, 125.5540542602539, 125.17489624023438, 126.69155883789062, 126.54573059082033,
            124.8540496826172, 125.04022216796876, 124.90303802490234, 125.3145751953125, 125.93189239501952, 125.91229248046876, 126.3924331665039,
            135.936279296875, 137.01414489746094, 138.55250549316406, 138.37612915039062, 137.9744110107422, 137.7098388671875, 137.89601135253906,
            134.89764404296875, 132.50677490234375, 134.2901153564453, 135.45614624023438, 138.35653686523438, 136.12245178222656, 133.2416534423828,
            131.41912841796875, 133.9863739013672, 134.30972290039062, 133.6630096435547, 133.73158264160156, 133.3102569580078, 131.3995361328125,
            132.50677490234375, 131.6151123046875, 135.66192626953125, 136.69078063964844, 136.37722778320312, 135.25039672851562, 133.83937072753906,
            132.61456298828125, 132.93789672851562, 134.1823272705078, 134.69186401367188, 132.9770965576172, 134.8094482421875, 137.64126586914062,
            136.28904724121094, 136.53399658203125, 137.85682678222656, 137.89601135253906, 138.17037963867188, 138.16058349609375, 139.1600341796875,
            138.64071655273438, 140.20848083496094, 140.37506103515625, 139.65975952148438, 139.45401000976562, 136.76918029785156, 136.05386352539062,
            136.38702392578125, 136.39683532714844, 135.7991180419922, 140.54164123535156, 142.51116943359375, 143.6575927734375, 142.7006072998047,
            141.28977966308594, 142.4046173095703, 143.20376586914062, 145.44332885742188, 146.16354370117188, 146.87388610839844, 147.76182556152344,
            151.3234100341797, 151.59967041015625, 152.18174743652344, 154.0562744140625, 152.8723602294922, 155.8814697265625, 154.71730041503906,
            155.64468383789062, 153.9280242919922, 154.80609130859375, 155.84201049804688, 155.07244873046875, 153.3163299560547, 154.2535858154297,
            153.9280242919922, 154.65809631347656, 153.78990173339844, 151.72792053222656, 151.50100708007812, 152.3790740966797, 152.6947784423828, 148.8076171875,
            148.1761932373047, 146.63711547851562, 146.97254943847656, 147.4757080078125, 144.4172821044922, 145.3939971923828, 145.25587463378906,
            145.0782928466797, 145.58145141601562, 146.76536560058594, 146.1734161376953, 144.36795043945312, 144.85137939453125, 143.2530975341797,
            143.01632690429688, 141.79293823242188, 141.90147399902344, 142.5230255126953, 144.3778076171875, 144.44688415527344, 147.24879455566406,
            146.814697265625, 147.1205291748047, 146.93309020996094, 146.31153869628906, 145.16708374023438, 143.77598571777344, 144.48634338378906,
            142.98670959472656, 143.8253173828125, 145.61105346679688, 143.0755157470703, 141.84226989746094, 140.79649353027344, 141.42791748046875,
            142.02423095703125, 144.21075439453125, 143.8827667236328, 144.7573699951172, 145.25430297851562, 144.91639709472656, 147.0929718017578,
            146.9438934326172, 146.6258544921875, 145.0157928466797, 144.3995819091797, 142.07391357421875, 140.13587951660156, 140.30482482910156,
            139.53955078125, 139.8973388671875, 134.85841369628906, 136.57781982421875, 138.207763671875, 138.0885009765625, 140.5532989501953, 142.1236114501953,
            143.19700622558594, 143.12742614746094, 143.8330841064453, 143.40570068359375, 145.5325927734375, 144.88658142089844, 147.5302734375, 148.8223114013672,
            150.52183532714844, 151.88343811035156, 152.35055541992188, 152.0325164794922, 152.3903045654297, 152.59901428222656, 152.2511749267578,
            152.59901428222656, 153.37425231933594, 155.1234588623047, 155.87879943847656, 157.02175903320312, 157.0018768310547, 155.35205078125, 155.8291015625,
            157.5485076904297, 158.1249542236328, 159.5362548828125, 160.07293701171875, 162.9849853515625, 164.21737670898438, 165.21124267578125,
            167.4176483154297, 165.529296875, 166.4734649658203, 166.3740692138672, 167.35801696777344, 168.36181640625, 169.25631713867188, 169.0575408935547,
            171.02540588378906, 170.27999877929688, 171.41000366210938, 172.27000427246094, 172.02000427246094, 170.66000366210938, 171.02000427246094,
            170.3000030517578, 169.0500030517578, 167.99000549316406, 167.08999633789062, 167.4199981689453, 170.30999755859375, 170.11000061035156,
            168.99000549316406, 170.5, 172.94000244140625, 172.27999877929688, 172.72999572753906, 176.27000427246094, 174.36000061035156, 173.72999572753906,
            174.72999572753906, 174.5, 175.10000610351562, 175.42999267578125, 174.8000030517578, 175.00999450683594]);
        let dates: Vec<&str> = vec!["2023-02-13", "2023-02-14", "2023-02-15", "2023-02-16", "2023-02-17", "2023-02-21", "2023-02-22", "2023-02-23", "2023-02-24",
            "2023-02-27", "2023-02-28", "2023-03-01", "2023-03-02", "2023-03-03", "2023-03-06", "2023-03-07", "2023-03-08", "2023-03-09", "2023-03-10", "2023-03-13",
            "2023-03-14", "2023-03-15", "2023-03-16", "2023-03-17", "2023-03-20", "2023-03-21", "2023-03-22", "2023-03-23", "2023-03-24", "2023-03-27", "2023-03-28",
            "2023-03-29", "2023-03-30", "2023-03-31", "2023-04-03", "2023-04-04", "2023-04-05", "2023-04-06", "2023-04-10", "2023-04-11", "2023-04-12", "2023-04-13",
            "2023-04-14", "2023-04-17", "2023-04-18", "2023-04-19", "2023-04-20", "2023-04-21", "2023-04-24", "2023-04-25", "2023-04-26", "2023-04-27", "2023-04-28",
            "2023-05-01", "2023-05-02", "2023-05-03", "2023-05-04", "2023-05-05", "2023-05-08", "2023-05-09", "2023-05-10", "2023-05-11", "2023-05-12", "2023-05-15",
            "2023-05-16", "2023-05-17", "2023-05-18", "2023-05-19", "2023-05-22", "2023-05-23", "2023-05-24", "2023-05-25", "2023-05-26", "2023-05-30", "2023-05-31",
            "2023-06-01", "2023-06-02", "2023-06-05", "2023-06-06", "2023-06-07", "2023-06-08", "2023-06-09", "2023-06-12", "2023-06-13", "2023-06-14", "2023-06-15",
            "2023-06-16", "2023-06-20", "2023-06-21", "2023-06-22", "2023-06-23", "2023-06-26", "2023-06-27", "2023-06-28", "2023-06-29", "2023-06-30", "2023-07-03",
            "2023-07-05", "2023-07-06", "2023-07-07", "2023-07-10", "2023-07-11", "2023-07-12", "2023-07-13", "2023-07-14", "2023-07-17", "2023-07-18", "2023-07-19",
            "2023-07-20", "2023-07-21", "2023-07-24", "2023-07-25", "2023-07-26", "2023-07-27", "2023-07-28", "2023-07-31", "2023-08-01", "2023-08-02", "2023-08-03",
            "2023-08-04", "2023-08-07", "2023-08-08", "2023-08-09", "2023-08-10", "2023-08-11", "2023-08-14", "2023-08-15", "2023-08-16", "2023-08-17", "2023-08-18",
            "2023-08-21", "2023-08-22", "2023-08-23", "2023-08-24", "2023-08-25", "2023-08-28", "2023-08-29", "2023-08-30", "2023-08-31", "2023-09-01", "2023-09-05",
            "2023-09-06", "2023-09-07", "2023-09-08", "2023-09-11", "2023-09-12", "2023-09-13", "2023-09-14", "2023-09-15", "2023-09-18", "2023-09-19", "2023-09-20",
            "2023-09-21", "2023-09-22", "2023-09-25", "2023-09-26", "2023-09-27", "2023-09-28", "2023-09-29", "2023-10-02", "2023-10-03", "2023-10-04", "2023-10-05",
            "2023-10-06", "2023-10-09", "2023-10-10", "2023-10-11", "2023-10-12", "2023-10-13", "2023-10-16", "2023-10-17", "2023-10-18", "2023-10-19", "2023-10-20",
            "2023-10-23", "2023-10-24", "2023-10-25", "2023-10-26", "2023-10-27", "2023-10-30", "2023-10-31", "2023-11-01", "2023-11-02", "2023-11-03", "2023-11-06",
            "2023-11-07", "2023-11-08", "2023-11-09", "2023-11-10", "2023-11-13", "2023-11-14", "2023-11-15", "2023-11-16", "2023-11-17", "2023-11-20", "2023-11-21",
            "2023-11-22", "2023-11-24", "2023-11-27", "2023-11-28", "2023-11-29", "2023-11-30", "2023-12-01", "2023-12-04", "2023-12-05", "2023-12-06", "2023-12-07",
            "2023-12-08", "2023-12-11", "2023-12-12", "2023-12-13", "2023-12-14", "2023-12-15", "2023-12-18", "2023-12-19", "2023-12-20", "2023-12-21", "2023-12-22",
            "2023-12-26", "2023-12-27", "2023-12-28", "2023-12-29", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10",
            "2024-01-11", "2024-01-12", "2024-01-16", "2024-01-17", "2024-01-18", "2024-01-19", "2024-01-22", "2024-01-23", "2024-01-24", "2024-01-25", "2024-01-26",
            "2024-01-29", "2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02", "2024-02-05", "2024-02-06", "2024-02-07", "2024-02-08", "2024-02-09"];
        let portfolio_data: HashMap<String, Array1<f64>> = HashMap::from([(String::from("BAC"), bac), (String::from("C"), c),
                                                                          (String::from("GS"), gs), (String::from("JPM"), jpm),]);
        (dates, portfolio_data)
    }

    /// # Description
    /// Loads samples of data.
    pub fn load_sample_data(&self) -> (Vec<&str>, HashMap<String, Array1<f64>>) {
        match self {
            SampleData::CAPM => { self.load_capm_data() },
            SampleData::Portfolio => { self.load_portfolio_data() },
        }
    }
}


#[cfg(test)]
mod tests {

    #[test]
    fn unit_test_sample_capm() -> () {
        use crate::utilities::sample_data::SampleData;
        let sample: SampleData = SampleData::CAPM;
        let (dates, capm_data) = sample.load_sample_data();
        assert_eq!(dates[2], String::from("2019-03-31"));
        assert_eq!(capm_data.get("RF").unwrap()[55], 0.0045);
        assert_eq!(capm_data.get("HML").unwrap()[37], 0.0308);
    }

    #[test]
    fn unit_test_sample_portfolio() -> () {
        use crate::utilities::sample_data::SampleData;
        let sample: SampleData = SampleData::Portfolio;
        let (dates, portfolio_data) = sample.load_sample_data();
        assert_eq!(dates[15], String::from("2023-03-07"));
        assert_eq!(portfolio_data.get("C").unwrap()[44], 47.88409423828125);
        assert_eq!(portfolio_data.get("JPM").unwrap()[23], 122.31652069091795);
    }
}