package edu.poly.student.controller;

import java.nio.file.Path;

import javax.servlet.ServletContext;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.validation.BindingResult;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.client.RestTemplate;

import edu.poly.student.model.Ai_model;

@Controller
@RequestMapping("AI")
@Validated
public class AINhanDien {
	
	@Autowired
	ServletContext application;

	@GetMapping("home")
	public String home(Model model) {
		model.addAttribute("form", new Ai_model());
		return "form/home";
	}

	@PostMapping("result")
	public String submit(@Validated @ModelAttribute("form") Ai_model form, BindingResult result, Model model) {
	    if (result.hasErrors()) {
	        return "form/home";
	    }
	    //để lấy đường dẫn thư mục gốc của ứng dụng.
	    String path = application.getRealPath("/");
	    System.out.println(path);

	    try {
	        if (!form.getImageFile().isEmpty()) {
	        	//form.getImageFile().getOriginalFilename() để lấy tên file ảnh gốc
	            form.setImageUrl(form.getImageFile().getOriginalFilename());

	            String filePath = path + "images/" + form.getImageUrl();
	            //để lưu file ảnh vào thư mục images.
	            form.getImageFile().transferTo(Path.of(filePath));
	            System.out.println(filePath);

	            // Gọi hàm xử lý ảnh Python
	            
	            //lấy dữ liệu của file ảnh dưới dạng một mảng các byte
	            byte[] content = form.getImageFile().getBytes();
	            //để lưu trữ thông tin về các header của phản hồi HTTP.
	            HttpHeaders headers = new HttpHeaders();
	            //Đặt kiểu dữ liệu của phản hồi là MediaType.IMAGE_JPEG, tức là định dạng ảnh JPEG.
	            headers.setContentType(MediaType.IMAGE_JPEG);
	            //HttpEntity chứa các byte của file ảnh và các thông tin về loại media của file đó.
	            HttpEntity<byte[]> requestEntity = new HttpEntity<>(content, headers);
	            //RestTemplate dùng để gửi request POST tới URL http://localhost:5000/process_image
	            RestTemplate restTemplate = new RestTemplate();
	            String url = "http://localhost:5000/process_image";
	            
	            ResponseEntity<String> response = restTemplate.exchange(url, HttpMethod.POST, requestEntity, String.class);
	            //response.getBody() chứa kết quả trả về từ server Flask
	            //String text = response.getBody();
	            // Chuyển đổi encoding sang UTF-8
	            byte[] bytes = response.getBody().getBytes("ISO-8859-1");
	            String utf8Text = new String(bytes, "UTF-8");
	            model.addAttribute("text", utf8Text);
	            System.out.println(utf8Text);
	        }
	    } catch (Exception e) {
	        e.printStackTrace();
	    }

	    return "form/result";
	}
}